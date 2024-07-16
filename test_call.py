
import json
import requests

def call_api(text):
    response = requests.post(
        "http://liacpc11.epfl.ch:3212/api/v1/actions",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"inputs": text})
    )
    return response.json()

string = "A solution of ((1S,2S)-1-{[(4-methoxymethyl-biphenyl-4-yl)-(2-pyridin-2-yl-cyclopropanecarbonyl)-amino]-methyl}-2-methyl-butyl)-carbamic acid tert-butyl ester (25 mg, 0.045 mmol) and dichloromethane (4 mL) was treated with a solution of HCl in dioxane (4 N, 0.5 mL) and the resulting reaction mixture was maintained at room temperature for 12 h. The reaction was then concentrated to dryness to afford (1R,2R)-2-pyridin-2-yl-cyclopropanecarboxylic acid ((2S,3S)-2-amino-3-methylpentyl)-(4'-methoxymethyl-biphenyl-4-yl)-amide (18 mg, 95% yield) as a white solid."

product = call_api(string)
print(product)
