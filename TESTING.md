#How to Test

Start the app: make up → view at http://localhost:8501

For each case below, set the sidebar toggles/controls, click Analyze, then record:

Verdict (BENIGN/MALICIOUS)

If MALICIOUS: Predicted Actor & Cluster ID

Notes (anything unexpected)

Tip: To see raw cluster output, temporarily add this line after clustering in app.py

#Test Cases

#1 Benign URL (baseline safe)

Inputs

URL Length: Normal

SSL Status: Trusted

Sub‑domain: One

Has Prefix/Suffix: ✗

Uses IP Address: ✗

Is Shortened: ✗

Has '@' Symbol: ✗

Abnormal URL: ✗

Political Keyword: ✗

Expected

Verdict: BENIGN

Attribution tab: not shown / disable


#2 Organized Cybercrime (noisy)

Inputs

URL Length: Long

SSL Status: None

Sub‑domain: Many

Has Prefix/Suffix: ✓

Uses IP Address: ✓

Is Shortened: ✓

Has '@' Symbol: ✓ (optional)

Abnormal URL: ✓

Political Keyword: ✗

Expected

Verdict: MALICIOUS

Predicted Actor: Organized Cybercrime (Cluster often 0)


#3 State‑Sponsored (subtle, crafted)

Inputs

URL Length: Normal or Long

SSL Status: Trusted

Sub‑domain: One or Many

Has Prefix/Suffix: ✓

Uses IP Address: ✗

Is Shortened: ✗

Has '@' Symbol: ✗

Abnormal URL: ✗/✓ (minor)

Political Keyword: ✗

Expected

Verdict: MALICIOUS

Predicted Actor: State‑Sponsored (Cluster often 1)


#4 Hacktivist (opportunistic signal)

Inputs

URL Length: Short/Normal

SSL Status: Suspicious or None

Sub‑domain: One

Has Prefix/Suffix: ✓/✗

Uses IP Address: ✗/✓ (mixed)

Is Shortened: ✗/✓ (mixed)

Has '@' Symbol: ✗/✓ (mixed)

Abnormal URL: ✓/✗ (mixed)

Political Keyword: ✓

Expected

Verdict: MALICIOUS

Predicted Actor: Hacktivist (Cluster often 2)

#Negative / Edge Tests

All toggles off but Abnormal ✓ → MALICIOUS with moderate score; actor depends on mix.

Conflicting signals (Trusted SSL + Shortened + IP ✓) → should still go MALICIOUS; actor likely Organized Cybercrime.

No clustering model present (delete models/threat_actor_profiler.pkl) → app should warn and continue without attribution.
