import requests
import json


def test_predict_endpoint():
    url = "https://udacity-census-bureau-project-new.onrender.com/predict"

    sample_payload = {
        "age": 39,
        "workclass": "State-gov",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = requests.post(url, json=sample_payload)

    # Assert the response code
    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"

    # Assert the response contains prediction
    response_json = response.json()
    assert (
        "prediction" in response_json
    ), "Response JSON does not contain 'prediction' key"
    assert response_json["prediction"] in [0, 1], "Prediction should be 0 or 1"


sample_data = {
    "age": 39,
    "workclass": "State-gov",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "hours-per-week": 40,
    "native-country": "United-States",
}


def test_api():
    try:
        # Send POST request
        response = requests.post(
            "https://udacity-census-bureau-project-new.onrender.com/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(sample_data),
        )

        # Get results
        status_code = response.status_code
        result = response.json() if status_code == 200 else response.text

        return {
            "status_code": status_code,
            "result": result,
            "success": status_code == 200,
        }

    except Exception as e:
        return {
            "status_code": None,
            "result": f"Request failed: {str(e)}",
            "success": False,
        }


# Execute test
if __name__ == "__main__":
    test_result = test_api()

    print(f"Status Code: {test_result['status_code']}")
    print(f"Prediction Result: {test_result['result']}")

    if test_result["success"]:
        print("\n✅ Request succeeded!")
    else:
        print("\n❌ Request failed!")
