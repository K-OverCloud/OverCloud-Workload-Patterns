# _*_ coding: utf-8 _*_
import os
import subprocess
import thread
import sys
import pymongo
import requests
import re
import datetime

from pymongo import MongoClient
from flask import request
from flask import Flask
import threading, logging, time
import multiprocessing
import msgpack
from kafka import KafkaProducer
from kafka.errors import KafkaError
from time import localtime, strftime
from pyowm import OWM


SELECTOR_mongo = 'edgex-mongo'
SELECTOR_kafka = 'kafka-broker-1'
app = Flask(__name__)
producer = KafkaProducer(bootstrap_servers=['kafka-broker-1:9092'])
#topic = 'Device1'


@app.route('/')
def index():
    connection = pymongo.MongoClient(SELECTOR_mongo, 27017)
    # connection = pymongo.MongoClient("localhost", 27017)
    db = connection.coredata

    #read_data = db.reading.find()

    read_data = db.reading.find().limit(4)

    str1 = str()
    for data in read_data:
        str1 = str1 + "{}:{}, ".format(data['name'], data['value'])
    print(str1)

    str2 = str1.split(',')
    device_name = str2[0]
    str4 = str2[1]
    str5 = str2[2]
    str6 = str2[3]
    str4 = str4.split(' ')
    time_val = str4[1]
    str5 = str5.split(' ')
    temperature = str5[1]
    str6 = str6.split(' ')
    humidity = str6[1]

    device_name2 = device_name.split(':')
    device_name = device_name2[1]

    producer.send(device_name, str1)
    db.reading.remove({})

    '''API_key = '5ae0b0ddfba53bb1acc0ec416e931177'  ## information for weather configure
    owm = OWM(API_key)
    obs = owm.weather_at_place('Gwangju')
    obs = owm.weather_at_coords(35.15972, 126.85306)
    
    w = obs.get_weather()
    weather_stat = w.get_status()
    Temperature = w.get_temperature(unit='celsius')['temp']
    Temperature="{0:0.2f}".format(Temperature)
    Humidity = w.get_humidity()
    Humidity= "{0:0.2f}".format(Humidity)
    date = strftime("%y.%m.%d-%H:%M:%S", localtime())
    weather_message = "weather:%s, time:%s, temperature:%s, humidity:%s," % (weather_stat, date, Temperature, Humidity)
    print(weather_message)
    producer.send("weather", str(weather_message))'''

    return 'connection is good'

if __name__ == '__main__':
   # app.debug = True
    app.run(host='0.0.0.0')




