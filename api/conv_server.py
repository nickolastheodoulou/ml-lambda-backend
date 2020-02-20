from flask import Flask, request
import fastai.vision as fastai

import boto3
from io import BytesIO

BUCKET_NAME = 'python-stream'
MODEL_FILE_NAME = 'classifier.pkl'
app = Flask(__name__)
S3 = boto3.client('s3', region_name='eu-west-1')

app = Flask(__name__)

#  Load model from S3 bucket
response = S3.get_object(Bucket=BUCKET_NAME, Key=MODEL_FILE_NAME)
# Load pickle model
model_str = response['Body'].read()
model_str = BytesIO(model_str)

CLASSIFIER = fastai.load_learner("models", model_str)


@app.route("/classify", methods=["POST", "OPTIONS"])  # define the API methods
def classify():
    files = request.files
    image = fastai.image.open_image(files['image'])  # load the image from the API call
    prediction = CLASSIFIER.predict(image)  # make a classification prediction from the call
    print(prediction)

    return {
        "brandPredictions": sorted(
            list(
                zip(
                    CLASSIFIER.data.classes,
                    [round(x, 4) for x in map(float, prediction[2])]
                )
            ),
            key=lambda p: p[1],
            reverse=True
        )  # data processed to be printed correctly
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
