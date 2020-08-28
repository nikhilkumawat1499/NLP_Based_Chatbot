# import files
from flask import Flask, render_template, request, jsonify
from nlp_chatbot import chat_tfidf
#from chatbot import bot
from flask_cors import CORS
import re
import json
from urllib.request import urlopen

url = 'http://ipinfo.io/json'
response = urlopen(url)
data = json.load(response)

IP=data['ip']
org=data['org']
city = data['city']
country=data['country']
region=data['region']

app = Flask(__name__)
CORS(app)

print ('Your IP details:\n ')
print ('IP : {4} \nRegion : {1} \nCountry : {2} \nCity : {3} \nOrg : {0}'.format(org,region,country,city,IP))


@app.route("/", methods=['GET'])
def index():    
    return render_template("index1.html") 
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    message = str(chat_tfidf(userText))    
    return message
if __name__ == "__main__":   
    app.run()
