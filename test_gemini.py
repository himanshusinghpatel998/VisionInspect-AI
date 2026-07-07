from ai_report import generate_report

report = generate_report(
    prediction="Defective",
    confidence=96.5,
    score=33.37
)

print(report)