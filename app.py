from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def root():
    return jsonify(message="📈 Forecasting service for github repos")

@app.route('/forecast/issues')
def forecast_issues():
    return jsonify(forecast="This will serve the issue forecast data.")

@app.route('/forecast/commits')
def forecast_commits():
    return jsonify(forecast="This will serve the commit forecast data.")

# Add more endpoints as needed

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)