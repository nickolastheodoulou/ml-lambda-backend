from flask import Flask, request, json
import boto3
from sklearn.externals import joblib
from io import BytesIO

BUCKET_NAME = 'python-stream'
MODEL_FILE_NAME = 'model.pkl'
app = Flask(__name__)
S3 = boto3.client('s3', region_name='eu-west-1')


@app.route('/', methods=['POST'])
def index():    
    # Parse request body for model input 
    body_dict = request.get_json(silent=True)    
    data = body_dict['data']     
    
    #  Load model from S3 bucket
    response = S3.get_object(Bucket=BUCKET_NAME, Key=MODEL_FILE_NAME)
    # Load pickle model
    model_str = response['Body'].read()
    model_str = BytesIO(model_str)
    model = joblib.load(model_str)

    # Make prediction
    prediction = model.predict(data).tolist()
    # Respond with prediction result
    result = {'prediction': prediction}    
   
    return json.dumps(result)


if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0')
