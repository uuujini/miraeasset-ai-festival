from transformers import pipeline

# 파이프라인 로드
pipe = pipeline("text-classification", model="yiyanghkust/finbert-esg-9-categories")

# 예시 텍스트
text = """
Apple announced significant progress towards its goal to decarbonize its value chain, including revealing that more than 320 suppliers – representing 95% of the company’s direct manufacturing spend – have now committed to use 100% renewable energy for Apple production by 2030, up from around 250 suppliers last year.

Emissions from the product manufacturing represents nearly two thirds of Apple’s carbon footprint, with electricity use as the single largest contributor. Addressing these emissions forms a major part of the company’s strategy to achieve its ambition to become carbon neutral across its entire business, manufacturing supply chain, and product life cycle by 2030, a goal set by the company in 2020.

In October 2022, Apple urged its supply chain to decarbonize their entire Apple-related Scope 1 and 2 emissions footprint, and notified suppliers that progress towards these goals would be one of the key criteria considered when awarding business. Since that time, the use of clean energy in Apple’s supply chain has grown rapidly, reaching 16.5 GW currently, up 20% over last year, and more than 55% higher than 2022.

Apple said that its supply chain generated more than 25.5 million MW of clean energy last year, avoiding over 18.5 million metric tons of carbon emissions.
"""

# 텍스트 분류 수행
result = pipe(text)
print(result)
