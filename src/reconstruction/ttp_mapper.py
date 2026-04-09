"""
src/reconstruction/ttp_mapper.py
─────────────────────────────────────────────────────────────────────────────
Maps each node in a reconstructed attack path to its MITRE ATT&CK TTP(s).

Two mapping strategies work in parallel and are merged:

  A. Feature-based mapping
     Each ProcessNode carries binary feature flags (has_encoded_cmd,
     has_download, has_credential, etc.) derived from the node feature
     vector. These flags map deterministically to TTP codes using a
     rule table with no heuristics beyond the feature extractor in
     graph/builder.py.

  B. Image-based mapping
     The process image name maps to known LOLBin TTPs from a static
     lookup table derived from the LOLBAS project.

The two strategies are unioned per node. The full session TTP set is
then ordered by MITRE kill-chain stage to produce the final kill chain.

Output per session
──────────────────
KillChain:
  session_id      str
  chain           list[ChainStep]  — one per TTP, ordered by stage
  summary         str              — one-line human-readable description
  mitre_url       str              — ATT&CK navigator deep-link

ChainStep:
  stage_num       int              — 1=Initial Access … 11=Exfiltration
  tactic          str              — ATT&CK tactic name
  technique_id    str              — e.g. "T1059.001"
  technique_name  str              — e.g. "PowerShell"
  node_idx        int              — which node in AttackPath triggered this
  process_image   str              — process that executed this technique
  evidence        list[str]        — human-readable evidence tokens
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import json

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TTP catalogue
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TTPEntry:
    technique_id:   str
    technique_name: str
    tactic:         str
    stage:          int       # kill-chain stage 1-11
    description:    str


TTP_CATALOGUE: dict[str, TTPEntry] = {
    # ── Initial Access ─────────────────────────────────────────────────────────
    "T1566.001": TTPEntry("T1566.001", "Spearphishing Attachment", "Initial Access",     1,
                          "Malicious document delivered via email attachment"),
    "T1566.002": TTPEntry("T1566.002", "Spearphishing Link",       "Initial Access",     1,
                          "Malicious link delivered via email"),
    "T1190":     TTPEntry("T1190",     "Exploit Public-Facing App","Initial Access",     1,
                          "Exploitation of internet-facing service"),
    # ── Execution ─────────────────────────────────────────────────────────────
    "T1059.001": TTPEntry("T1059.001", "PowerShell",               "Execution",          2,
                          "Adversary used PowerShell for code execution"),
    "T1059.003": TTPEntry("T1059.003", "Windows Command Shell",    "Execution",          2,
                          "cmd.exe used for command execution"),
    "T1204.002": TTPEntry("T1204.002", "Malicious File",           "Execution",          2,
                          "User executed a malicious file"),
    "T1218":     TTPEntry("T1218",     "System Binary Proxy Exec", "Execution",          2,
                          "LOLBin used to proxy execution and evade defenses"),
    "T1218.011": TTPEntry("T1218.011", "Rundll32",                 "Execution",          2,
                          "rundll32.exe used to execute malicious DLL"),
    "T1218.005": TTPEntry("T1218.005", "Mshta",                    "Execution",          2,
                          "mshta.exe used to execute malicious HTA"),
    "T1218.010": TTPEntry("T1218.010", "Regsvr32",                 "Execution",          2,
                          "regsvr32.exe used for proxy execution"),
    # ── Persistence ───────────────────────────────────────────────────────────
    "T1053.005": TTPEntry("T1053.005", "Scheduled Task",           "Persistence",        3,
                          "Scheduled task created for persistence"),
    "T1547.001": TTPEntry("T1547.001", "Registry Run Keys",        "Persistence",        3,
                          "Registry autorun key modified for persistence"),
    "T1546.003": TTPEntry("T1546.003", "WMI Event Subscription",   "Persistence",        3,
                          "WMI event subscription for persistent execution"),
    # ── Privilege Escalation ───────────────────────────────────────────────────
    "T1055":     TTPEntry("T1055",     "Process Injection",        "Privilege Escalation", 4,
                          "Code injected into another process"),
    "T1548.002": TTPEntry("T1548.002", "Bypass UAC",               "Privilege Escalation", 4,
                          "UAC bypass technique used to escalate privileges"),
    # ── Defense Evasion ───────────────────────────────────────────────────────
    "T1027":     TTPEntry("T1027",     "Obfuscated Files/Info",    "Defense Evasion",    5,
                          "Command or payload was obfuscated (Base64/encoding)"),
    "T1027.010": TTPEntry("T1027.010", "Command Obfuscation",      "Defense Evasion",    5,
                          "PowerShell command was obfuscated using encoding or concatenation"),
    "T1562.001": TTPEntry("T1562.001", "Disable Security Tools",   "Defense Evasion",    5,
                          "Security tool or policy was disabled"),
    "T1070.004": TTPEntry("T1070.004", "File Deletion",            "Defense Evasion",    5,
                          "Files deleted to remove forensic evidence"),
    # ── Credential Access ─────────────────────────────────────────────────────
    "T1003.001": TTPEntry("T1003.001", "LSASS Memory",             "Credential Access",  6,
                          "LSASS process accessed to dump credentials"),
    "T1552.001": TTPEntry("T1552.001", "Credentials in Files",     "Credential Access",  6,
                          "Credentials found in files on disk"),
    "T1555":     TTPEntry("T1555",     "Credentials from PWD Store","Credential Access", 6,
                          "Credentials extracted from password manager or browser"),
    # ── Discovery ─────────────────────────────────────────────────────────────
    "T1082":     TTPEntry("T1082",     "System Info Discovery",    "Discovery",          7,
                          "System and OS information gathered"),
    "T1069.001": TTPEntry("T1069.001", "Local Groups",             "Discovery",          7,
                          "Local user group membership enumerated"),
    "T1083":     TTPEntry("T1083",     "File & Dir Discovery",     "Discovery",          7,
                          "File system structure enumerated"),
    "T1016":     TTPEntry("T1016",     "Network Config Discovery", "Discovery",          7,
                          "Network configuration information gathered"),
    # ── Lateral Movement ──────────────────────────────────────────────────────
    "T1021.002": TTPEntry("T1021.002", "SMB/Windows Admin Shares", "Lateral Movement",  8,
                          "Lateral movement via SMB or admin shares"),
    "T1021.006": TTPEntry("T1021.006", "WinRM",                    "Lateral Movement",  8,
                          "WinRM used for remote command execution"),
    "T1570":     TTPEntry("T1570",     "Lateral Tool Transfer",    "Lateral Movement",  8,
                          "Tool or payload transferred to remote host"),
    # ── Collection ────────────────────────────────────────────────────────────
    "T1005":     TTPEntry("T1005",     "Data from Local System",   "Collection",         9,
                          "Data collected from local file system"),
    "T1074.001": TTPEntry("T1074.001", "Local Data Staging",       "Collection",         9,
                          "Data staged locally before exfiltration"),
    # ── Command & Control ─────────────────────────────────────────────────────
    "T1071.001": TTPEntry("T1071.001", "Web Protocols",            "Command & Control", 10,
                          "HTTP/HTTPS used for C2 communication"),
    "T1105":     TTPEntry("T1105",     "Ingress Tool Transfer",    "Command & Control", 10,
                          "Tool or payload downloaded from remote server"),
    "T1095":     TTPEntry("T1095",     "Non-Application Layer",    "Command & Control", 10,
                          "Non-standard protocol used for C2"),
    # ── Exfiltration ──────────────────────────────────────────────────────────
    "T1041":     TTPEntry("T1041",     "Exfiltration Over C2",     "Exfiltration",      11,
                          "Data exfiltrated over the C2 channel"),
    "T1567.002": TTPEntry("T1567.002", "Exfiltration to Cloud",    "Exfiltration",      11,
                          "Data exfiltrated to cloud storage service"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Image → TTP lookup (LOLBAS-derived)
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_TO_TTPS: dict[str, list[str]] = {
    "powershell.exe":  ["T1059.001"],
    "pwsh.exe":        ["T1059.001"],
    "cmd.exe":         ["T1059.003"],
    "certutil.exe":    ["T1218", "T1105"],
    "mshta.exe":       ["T1218.005"],
    "regsvr32.exe":    ["T1218.010"],
    "rundll32.exe":    ["T1218.011"],
    "wscript.exe":     ["T1218"],
    "cscript.exe":     ["T1218"],
    "msiexec.exe":     ["T1218"],
    "installutil.exe": ["T1218"],
    "regasm.exe":      ["T1218"],
    "msbuild.exe":     ["T1218"],
    "bitsadmin.exe":   ["T1105"],
    "wmic.exe":        ["T1546.003", "T1082"],
    "schtasks.exe":    ["T1053.005"],
    "reg.exe":         ["T1547.001"],
    "net.exe":         ["T1069.001", "T1021.002"],
    "net1.exe":        ["T1069.001"],
    "psexec.exe":      ["T1021.002"],
    "winrm.cmd":       ["T1021.006"],
    "mimikatz.exe":    ["T1003.001"],
    "procdump.exe":    ["T1003.001"],
    "lsass.exe":       ["T1003.001"],
    "esentutl.exe":    ["T1218"],
    "findstr.exe":     ["T1218"],
    "diskshadow.exe":  ["T1218"],
    "odbcconf.exe":    ["T1218"],
    "expand.exe":      ["T1218"],
    "xcopy.exe":       ["T1570"],
    "robocopy.exe":    ["T1570"],
    "curl.exe":        ["T1105", "T1071.001"],
    "wget.exe":        ["T1105", "T1071.001"],
    "ftp.exe":         ["T1105"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature flag → TTP mapping
# ─────────────────────────────────────────────────────────────────────────────

def _ttps_from_node_features(node) -> list[tuple[str, list[str]]]:
    """
    Return list of (ttp_code, evidence_tokens) from a ProcessNode's
    feature flags. Each flag maps to one or more TTPs.
    """
    result = []

    if node.has_encoded_cmd:
        result.append(("T1027.010", ["encoded command detected (-enc/-EncodedCommand)"]))

    if node.has_download:
        result.append(("T1105", ["download pattern (DownloadString/IWR/WebClient)"]))
        result.append(("T1071.001", ["HTTP/S channel implied by download"]))

    if node.has_credential:
        result.append(("T1003.001", ["credential access pattern (mimikatz/lsass/sekurlsa)"]))

    if node.has_lateral:
        result.append(("T1021.002", ["lateral movement pattern (psexec/Enter-PSSession)"]))
        result.append(("T1021.006", ["WinRM lateral movement detected"]))

    if node.has_network and not node.has_download:
        result.append(("T1071.001", ["outbound network connection observed"]))

    if node.is_lolbin:
        result.append(("T1218", [f"LOLBin execution: {node.image}"]))

    if node.integrity_level in ("High", "System") and node.depth > 1:
        result.append(("T1548.002", ["elevated integrity level on child process"]))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Chain step output object
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChainStep:
    stage_num:      int
    tactic:         str
    technique_id:   str
    technique_name: str
    node_idx:       int
    process_image:  str
    evidence:       list[str] = field(default_factory=list)
    description:    str = ""


@dataclass
class KillChain:
    session_id:  str
    p_malicious: float
    chain:       list[ChainStep]
    n_hops:      int
    root_image:  str
    leaf_image:  str
    summary:     str
    mitre_url:   str
    raw_ttps:    list[str]      # all unique TTP codes in chain

    def to_dict(self) -> dict:
        return {
            "session_id":  self.session_id,
            "p_malicious": round(self.p_malicious, 4),
            "n_hops":      self.n_hops,
            "root_image":  self.root_image,
            "leaf_image":  self.leaf_image,
            "summary":     self.summary,
            "mitre_url":   self.mitre_url,
            "raw_ttps":    self.raw_ttps,
            "chain": [
                {
                    "stage":          s.stage_num,
                    "tactic":         s.tactic,
                    "technique_id":   s.technique_id,
                    "technique_name": s.technique_name,
                    "process":        s.process_image,
                    "evidence":       s.evidence,
                    "description":    s.description,
                }
                for s in self.chain
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# TTP mapper — main entry point
# ─────────────────────────────────────────────────────────────────────────────

def map_attack_path_to_ttps(attack_path) -> KillChain:
    """
    Given an AttackPath, produce a fully annotated KillChain.

    Parameters
    ──────────
    attack_path : AttackPath (from graph_traversal.py)

    Returns
    ───────
    KillChain with ordered ChainStep list
    """
    all_steps: list[ChainStep] = []

    # Root node — initial access heuristic
    if attack_path.nodes:
        root = attack_path.nodes[0]
        parent_triggers = []
        # If root's parent was an Office app or browser — phishing entry
        # We check feature: is the first node spawned from an unusual parent?
        # (We encode this as depth=0 and is_powershell=True with
        #  parent_is_office or parent_is_browser in tabular features;
        #  here we use image-based heuristic on root image name)
        if root.image in {"winword.exe", "excel.exe", "outlook.exe",
                          "powerpnt.exe", "chrome.exe", "firefox.exe",
                          "msedge.exe", "iexplore.exe"}:
            all_steps.append(ChainStep(
                stage_num      = 1,
                tactic         = "Initial Access",
                technique_id   = "T1566.001",
                technique_name = "Spearphishing Attachment",
                node_idx       = root.node_idx,
                process_image  = root.image,
                evidence       = [f"Session root process is {root.image} — typical phishing entry point"],
                description    = TTP_CATALOGUE["T1566.001"].description,
            ))

    # Walk each node in the attack path
    seen_ttps: set[str] = set()

    for node in attack_path.nodes:
        # Strategy A: feature flags
        feat_ttps = _ttps_from_node_features(node)
        for ttp_code, evidence in feat_ttps:
            if ttp_code not in TTP_CATALOGUE or ttp_code in seen_ttps:
                continue
            entry = TTP_CATALOGUE[ttp_code]
            all_steps.append(ChainStep(
                stage_num      = entry.stage,
                tactic         = entry.tactic,
                technique_id   = entry.technique_id,
                technique_name = entry.technique_name,
                node_idx       = node.node_idx,
                process_image  = node.image,
                evidence       = evidence,
                description    = entry.description,
            ))
            seen_ttps.add(ttp_code)

        # Strategy B: image lookup
        img_lower = node.image.lower()
        for ttp_code in IMAGE_TO_TTPS.get(img_lower, []):
            if ttp_code not in TTP_CATALOGUE or ttp_code in seen_ttps:
                continue
            entry = TTP_CATALOGUE[ttp_code]
            all_steps.append(ChainStep(
                stage_num      = entry.stage,
                tactic         = entry.tactic,
                technique_id   = entry.technique_id,
                technique_name = entry.technique_name,
                node_idx       = node.node_idx,
                process_image  = node.image,
                evidence       = [f"Process image '{node.image}' matches known LOLBin/technique"],
                description    = entry.description,
            ))
            seen_ttps.add(ttp_code)

    # Sort by kill-chain stage
    all_steps.sort(key=lambda s: (s.stage_num, s.technique_id))

    # Build ATT&CK Navigator deep-link for these techniques
    raw_ttps = sorted(seen_ttps, key=lambda t: TTP_CATALOGUE.get(t, TTPEntry(t,"","",-1,"")).stage)
    nav_url  = _build_navigator_url(raw_ttps)

    # Build one-line summary
    summary = _build_summary(attack_path, all_steps)

    return KillChain(
        session_id  = attack_path.session_id,
        p_malicious = attack_path.p_malicious,
        chain       = all_steps,
        n_hops      = attack_path.n_hops,
        root_image  = attack_path.root_image,
        leaf_image  = attack_path.leaf_image,
        summary     = summary,
        mitre_url   = nav_url,
        raw_ttps    = raw_ttps,
    )


def _build_summary(attack_path, steps: list[ChainStep]) -> str:
    """One-line human-readable description of the attack chain."""
    tactics = []
    seen = set()
    for s in steps:
        if s.tactic not in seen:
            tactics.append(s.tactic)
            seen.add(s.tactic)

    tactic_str = " → ".join(tactics) if tactics else "Unknown"
    ttps_str   = ", ".join(s.technique_id for s in steps[:4])
    extra      = f" (+{len(steps)-4} more)" if len(steps) > 4 else ""

    return (
        f"PowerShell LOLBin attack: {tactic_str}. "
        f"Chain: {attack_path.root_image} → {attack_path.leaf_image} "
        f"({attack_path.n_hops} hops). "
        f"TTPs: {ttps_str}{extra}. "
        f"Confidence: {attack_path.p_malicious:.0%}"
    )


def _build_navigator_url(ttp_codes: list[str]) -> str:
    """
    Build a deep-link URL to the ATT&CK Navigator pre-highlighting
    the detected techniques. The link opens a layer with all detected
    TTPs coloured red — useful for the paper's case study figures.
    """
    if not ttp_codes:
        return "https://attack.mitre.org/"

    # Build a minimal Navigator layer JSON
    techniques = [
        {"techniqueID": t, "color": "#D85A30", "comment": "Detected by LOLBin system"}
        for t in ttp_codes
    ]
    layer = {
        "name":        "LOLBin Detection Result",
        "versions":    {"attack": "14", "navigator": "4.9"},
        "domain":      "enterprise-attack",
        "techniques":  techniques,
    }
    import urllib.parse
    layer_json = json.dumps(layer, separators=(",", ":"))
    encoded    = urllib.parse.quote(layer_json)
    return f"https://mitre-attack.github.io/attack-navigator/#layerURL=data:text/json;charset=utf-8,{encoded}"
