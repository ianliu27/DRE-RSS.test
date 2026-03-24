import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


N_A_STRINGS = {"", "n/a", "na", "none", "null", "undefined"}


def _is_blank_text(v: Any) -> bool:
    if v is None:
        return True
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return True
    return s.strip().lower() in N_A_STRINGS


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_term(normalized_text: str, normalized_term: str) -> bool:
    if not normalized_term:
        return False
    # Use "token-ish" boundaries for short alnum terms to avoid substring noise.
    if re.fullmatch(r"[a-z0-9]+", normalized_term) and len(normalized_term) <= 4:
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])")
        return bool(pat.search(normalized_text))
    return normalized_term in normalized_text


# ---------------------------------------------------------------------------
# Maintenance context override
# ---------------------------------------------------------------------------
# When a bid contains both AVAC and Manutenção signals, the word-level signals
# alone cannot distinguish "installing HVAC" from "maintaining HVAC".  The
# presence of any of these maintenance-context phrases is a strong indicator
# that the contract is a maintenance/service contract even when it also
# mentions HVAC-specific vocabulary.
#
# Effect: if the normalized text contains any of these phrases AND the
# combined send_to union includes both "AVAC" and "Manutenção", then "AVAC"
# is dropped from the final routing decision so the bid goes only to the
# Manutenção team.  If the text has no maintenance context, AVAC still wins
# (preserving the original priority order).
# ---------------------------------------------------------------------------

_MAINTENANCE_CONTEXT_PHRASES: List[str] = [
    "manutenção",
    "manutencao",
    "serviços de manutenção",
    "servicos de manutencao",
    "contrato de manutenção",
    "contrato de manutencao",
    "prestação de serviços de manutenção",
    "prestacao de servicos de manutencao",
    "manutenção preventiva",
    "manutencao preventiva",
    "manutenção corretiva",
    "manutencao corretiva",
    "manutenção e operação",
    "manutencao e operacao",
    "operação e manutenção",
    "operacao e manutencao",
    "assistência técnica",
    "assistencia tecnica",
    "suporte e manutenção",
    "suporte e manutencao",
    "higienização",
    "higienizacao",
    "reparação",
    "reparacao",
    "filtros para",
    "filtros de ar",
    "limpeza de condutas",
    "limpeza e higienização",
    "limpeza e higienizacao",
]

# Pre-compile once at import time for efficiency
_MAINTENANCE_CONTEXT_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"(?<![a-z])" + re.escape(
            unicodedata.normalize("NFD", phrase.lower()).replace(
                "\u0300", "").replace("\u0301", "").replace(
                "\u0302", "").replace("\u0303", "").replace(
                "\u0327", "").replace("\u0328", "")
        ) + r"(?![a-z])"
    )
    for phrase in _MAINTENANCE_CONTEXT_PHRASES
]

# Simpler normalisation used only for the phrase list above
def _norm_phrase(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip()

_MAINTENANCE_CONTEXT_COMPILED: List[re.Pattern] = [
    re.compile(r"(?<![a-z])" + re.escape(_norm_phrase(p)) + r"(?![a-z])")
    for p in _MAINTENANCE_CONTEXT_PHRASES
]


def _has_maintenance_context(normalized_text: str) -> bool:
    """
    Return True if the normalised bid text contains at least one maintenance-
    context phrase.  Used to resolve AVAC vs Manutenção ambiguity.
    """
    return any(pat.search(normalized_text) for pat in _MAINTENANCE_CONTEXT_COMPILED)


def _resolve_send_to(send_to_union: Set[str], normalized_text: str) -> List[str]:
    """
    Apply the maintenance-context override and return the final sorted send_to
    list.

    Rules (in priority order):
    1. If both 'AVAC' and 'Manutenção' are present AND the text contains a
       maintenance-context phrase → drop 'AVAC', keep 'Manutenção' (the bid
       is a maintenance contract that happens to mention HVAC vocabulary).
    2. Otherwise return the union as-is (sorted for determinism).
    """
    if "AVAC" in send_to_union and "Manutenção" in send_to_union:
        if _has_maintenance_context(normalized_text):
            send_to_union = send_to_union - {"AVAC"}
    return sorted(send_to_union)


def _select_search_text(proc: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (field_name, text) according to priority:
    detalhes_completos, else descricao + designacao_contrato.
    """
    detalhes = proc.get("detalhes_completos")
    if isinstance(detalhes, str) and not _is_blank_text(detalhes):
        return "detalhes_completos", detalhes

    descricao = proc.get("descricao")
    designacao = proc.get("designacao_contrato")

    parts: List[str] = []
    if isinstance(descricao, str) and not _is_blank_text(descricao):
        parts.append(descricao)
    if isinstance(designacao, str) and not _is_blank_text(designacao):
        parts.append(designacao)

    return "descricao+designacao_contrato", "\n".join(parts).strip()


CPV_FROM_TEXT_RE = re.compile(
    r"Vocabul[aá]rio\s+Principal:\s*([0-9]{4,10})",
    flags=re.IGNORECASE,
)


def _extract_cpvs(proc: Dict[str, Any], source_text: str) -> List[str]:
    """
    Extract CPV(s) from:
    - vocabulario_principal field, if present
    - else from occurrences of 'Vocabulário Principal:' in text
    Returns a de-duplicated list preserving discovery order.
    """
    seen: Set[str] = set()
    out: List[str] = []

    if "vocabulario_principal" in proc:
        v = proc.get("vocabulario_principal")
        candidates: List[Any]
        if isinstance(v, list):
            candidates = v
        else:
            candidates = [v]
        for c in candidates:
            if c is None:
                continue
            if isinstance(c, (int, float)):
                s = str(int(c))
            elif isinstance(c, str):
                s = c.strip()
            else:
                continue
            if not s or not re.fullmatch(r"[0-9]{4,10}", s):
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        if out:
            return out

    if isinstance(source_text, str) and source_text:
        for m in CPV_FROM_TEXT_RE.finditer(source_text):
            s = (m.group(1) or "").strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)

    return out


def _cpv_prefix(code: str, min_len: int = 2) -> Optional[str]:
    """
    Derive the significant prefix from a CPV filter code by stripping trailing
    zeros.  E.g. '45000000' -> '45', '45331000' -> '45331', '39700000' -> '397'.
    Returns None only when the digits string itself is shorter than *min_len*.
    """
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) < min_len:
        return None
    stripped = digits.rstrip("0")
    if len(stripped) < min_len:
        return digits[:min_len]
    return stripped


@dataclass(frozen=True)
class CpvMapItem:
    code: str
    prefix: str
    send_to: List[str]
    category: Optional[str]


@dataclass(frozen=True)
class TermMapItem:
    term: str
    normalized_term: str
    send_to: List[str]
    category: Optional[str]


def _load_json_object(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}; got {type(obj).__name__}")
    return obj


def load_cpv_map(path: str) -> Dict[str, CpvMapItem]:
    raw = _load_json_object(path)
    out: Dict[str, CpvMapItem] = {}
    for k, v in raw.items():
        code = str(k).strip()
        if not re.fullmatch(r"[0-9]{4,10}", code):
            continue
        if not isinstance(v, dict):
            continue
        send_to = v.get("send_to")
        if not isinstance(send_to, list) or not all(isinstance(x, str) and x.strip() for x in send_to):
            continue
        category = v.get("category") if isinstance(v.get("category"), str) else None
        prefix = _cpv_prefix(code, min_len=2)
        if prefix is None:
            continue
        out[code] = CpvMapItem(code=code, prefix=prefix, send_to=[x.strip() for x in send_to], category=category)
    return out


def load_semantic_map(path: str) -> Dict[str, TermMapItem]:
    raw = _load_json_object(path)
    out: Dict[str, TermMapItem] = {}
    for term, v in raw.items():
        if not isinstance(term, str):
            continue
        t = term.strip()
        if not t:
            continue
        if not isinstance(v, dict):
            continue
        send_to = v.get("send_to")
        if not isinstance(send_to, list) or not all(isinstance(x, str) and x.strip() for x in send_to):
            continue
        category = v.get("category") if isinstance(v.get("category"), str) else None
        out[t] = TermMapItem(
            term=t,
            normalized_term=_normalize_text(t),
            send_to=[x.strip() for x in send_to],
            category=category,
        )
    return out


def _mk_reason(reason_type: str, value: str, *, category: Optional[str], send_to: List[str]) -> Dict[str, Any]:
    return {"type": reason_type, "value": value, "category": category, "send_to": send_to}


def _match_cpvs(
    proc_cpvs: Sequence[str],
    cpv_map: Dict[str, CpvMapItem],
) -> List[Dict[str, Any]]:
    if not proc_cpvs or not cpv_map:
        return []

    matched: List[Dict[str, Any]] = []

    for proc_cpv in proc_cpvs:
        for _fcode, item in cpv_map.items():
            if proc_cpv == item.code:
                matched.append(
                    _mk_reason("cpv_exact", proc_cpv, category=item.category, send_to=item.send_to)
                )
            elif proc_cpv.startswith(item.prefix):
                matched.append(
                    _mk_reason("cpv_prefix", item.prefix, category=item.category, send_to=item.send_to)
                )

    # de-dupe by (type, value)
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for m in matched:
        key = (str(m.get("type")), str(m.get("value")))
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _match_terms(
    normalized_text: str,
    term_map: Dict[str, TermMapItem],
) -> List[Dict[str, Any]]:
    if not normalized_text or not term_map:
        return []

    matched: List[Dict[str, Any]] = []
    for _term_key, item in term_map.items():
        if _contains_term(normalized_text, item.normalized_term):
            matched.append(_mk_reason("term", item.term, category=item.category, send_to=item.send_to))

    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for m in matched:
        key = (str(m.get("type")), str(m.get("value")))
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _filter_base_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_filter(
    source_json_path: str,
    cpv_map_path: str,
    semantic_map_path: str,
    out_dir: str = os.path.join("data", "filtered"),
) -> Dict[str, Any]:
    with open(source_json_path, "r", encoding="utf-8") as f:
        procedures = json.load(f)
    if not isinstance(procedures, list):
        raise ValueError(f"Source JSON must be a list; got {type(procedures).__name__}")

    cpv_map = load_cpv_map(cpv_map_path)
    term_map = load_semantic_map(semantic_map_path)

    matched_records: List[Dict[str, Any]] = []
    seen_links: Set[str] = set()
    duplicates_removed = 0

    reasons_counter = {"cpv_exact": 0, "cpv_prefix": 0, "term": 0}
    specialty_counter: Dict[str, int] = {}
    overlap_counter: Dict[str, int] = {}
    maintenance_override_count = 0
    considered = 0

    for proc in procedures:
        if not isinstance(proc, dict):
            continue
        considered += 1
        link = proc.get("link")
        if not isinstance(link, str) or not link.strip():
            # Still allow filtering, but dedupe can't be guaranteed; skip for safety.
            continue

        source_field, search_text = _select_search_text(proc)
        normalized_text = _normalize_text(search_text) if search_text else ""

        proc_cpvs = _extract_cpvs(proc, search_text)
        cpv_hits = _match_cpvs(proc_cpvs, cpv_map)
        term_hits = _match_terms(normalized_text, term_map)

        matched_by: List[Dict[str, Any]] = cpv_hits + term_hits
        if not matched_by:
            continue

        if link in seen_links:
            duplicates_removed += 1
            continue
        seen_links.add(link)

        send_to_union: Set[str] = set()
        for r in matched_by:
            rtype = r.get("type")
            if rtype in reasons_counter:
                reasons_counter[rtype] += 1
            st = r.get("send_to")
            if isinstance(st, list):
                for x in st:
                    if isinstance(x, str) and x.strip():
                        send_to_union.add(x.strip())

        send_to_list = _resolve_send_to(send_to_union, normalized_text)
        # Track whether the maintenance override fired for this record
        if "AVAC" in send_to_union and "Manutenção" in send_to_union and _has_maintenance_context(normalized_text):
            maintenance_override_count += 1
        for s in send_to_list:
            specialty_counter[s] = specialty_counter.get(s, 0) + 1
        if send_to_list:
            overlap_key = "+".join(send_to_list)
            overlap_counter[overlap_key] = overlap_counter.get(overlap_key, 0) + 1

        enriched = dict(proc)
        enriched["filter_match"] = {
            "send_to": send_to_list,
            "matched_by": matched_by,
            "source_text_field": source_field,
            "cpv_extracted": proc_cpvs,
        }
        matched_records.append(enriched)

    source_base = _filter_base_name(source_json_path)
    out_path = os.path.join(out_dir, f"{source_base}_classified.json")
    report_path = os.path.join(out_dir, f"{source_base}_classified.report.json")

    _write_json(out_path, matched_records)
    report = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_json_path": source_json_path,
        "cpv_map_path": cpv_map_path,
        "semantic_map_path": semantic_map_path,
        "output_json_path": out_path,
        "total_input_records": len(procedures) if isinstance(procedures, list) else None,
        "total_considered_dict_records": considered,
        "total_matched_output_records": len(matched_records),
        "duplicates_removed_by_link": duplicates_removed,
        "match_reason_tallies": reasons_counter,
        "maintenance_override_applied": maintenance_override_count,
        "specialty_counts": specialty_counter,
        "overlap_counts": overlap_counter,
        "filter_stats": {"cpv_map_items": len(cpv_map), "semantic_map_items": len(term_map)},
    }
    _write_json(report_path, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Classify procedures by CPV + semantic maps (send_to).")
    parser.add_argument("--source", required=True, help="Path to source procedures JSON (list).")
    parser.add_argument("--cpv-map", required=True, help="Path to CPV map JSON (code -> {send_to, category}).")
    parser.add_argument("--semantic-map", required=True, help="Path to semantic map JSON (term -> {send_to, category}).")
    parser.add_argument("--out-dir", default=os.path.join("data", "filtered"), help="Output directory.")

    args = parser.parse_args(argv)

    try:
        report = run_filter(args.source, args.cpv_map, args.semantic_map, args.out_dir)
        print(
            f"[CLASSIFY] ok source={_filter_base_name(args.source)} matched={report['total_matched_output_records']} "
            f"out={report['output_json_path']}"
        )
        return 0
    except Exception as e:
        print(f"[CLASSIFY] failed: {e!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
