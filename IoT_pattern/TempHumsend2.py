#coding:utf-8  ##한글 처리

import requests, json
import Adafruit_DHT as dht
import time
import requests, json
import Adafruit_DHT as dht
import time
import os
import sys
import subprocess
from time import localtime, strftime


class EdgeX:
    def __init__(self, ip='IoT-Cloud Hub IP address'):
        self.ip = ip
        self.port = 48080
        self.metaIp = ip
        self.metaPort = 48081
        self.headers = {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*','Access-Control-Allow-Headers': '*', 'Access-Control-Allow-Methods' : 'POST, GET, OPTIONS'}

    def DataTemplate(self):
        deviceTemplate = '{"name":"device","formatting":"%s"}'
        tempTemplate = '{"name":"temperature","min":"-40","max":"140","type":"F","uomLabel":"degree cel","defaultValue":"0","formatting":"%s","labels":["temp","hvac"]}'
        humidTemplate = '{"name":"humidity","min":"0","max":"100","type":"F","uomLabel":"per","defaultValue":"0","formatting":"%s","labels":["humidity","hvac"]}'
        dateTemplate = '{"name":"time","formatting":"%s","labels":["time","YYYYMMDD HHMMSS"]}'
        url = 'http://%s:%d/api/v1/valuedescriptor' % (self.ip, self.port)
        response = requests.post(url, data=deviceTemplate, headers=self.headers)
        response = requests.post(url, data=tempTemplate, headers=self.headers)
        response = requests.post(url, data=humidTemplate, headers=self.headers)
        response = requests.post(url, data=dateTemplate, headers=self.headers)


    def createDevice(self, deviceName):
        print('Creating addressable for ', deviceName)
        addresableData = '{"origin":1471806386919,"name":"%s","protocol":"HTTP","address":"","port":"161","path":"","publisher":"none","user":"none","password":"none","topic":"none"}' % (
        deviceName,)
        url = 'http://%s:%d/api/v1/addressable' % (self.metaIp, self.metaPort)
        response = requests.post(url, data=addresableData, headers=self.headers)
        if response.status_code != 200:
            print(response.status_code)
            print(response.content)
        else:
            print('OK')

        print('Creating addressable for service')
        serviceAddresableData = '{"origin":1471806386919,"name":"%s-address","protocol":"HTTP","address":"","port":"49989","path":"","publisher":"none","user":"none","password":"none","topic":"none"}'%(deviceName,)
        url = 'http://%s:%d/api/v1/addressable' % (self.metaIp, self.metaPort)
        response = requests.post(url, data=serviceAddresableData, headers=self.headers)
        if response.status_code != 200:
            print(response.status_code)
            print(response.content)
        else:
            print('OK')

        print('Creating service')
        serviceData = '{"origin":1471806386919,"name":"edgex-%s","description":"temperature service for rooms","lastConnected":0,"lastReported":0,"labels":["snmp","rtu","io"],"adminState":"unlocked","operatingState":"enabled","addressable":{"name":"%s-address"}}'%(deviceName, deviceName,)
        url = 'http://%s:%d/api/v1/deviceservice' % (self.metaIp, self.metaPort)
        response = requests.post(url, data=serviceData, headers=self.headers)
        if response.status_code != 200:
            print(response.status_code)
            print(response.content)
        else:
            print('OK')




edgex = EdgeX()
edgex.DataTemplate()
edgex.createDevice("Device1")
edgex.createDevice("Device2")

while True:
    date = strftime("%y.%m.%d-%H:%M:%S", localtime())
    h, t = dht.read_retry(dht.DHT22, 4)
    if t <= 35:
        print 'Temp={0:0.1f}*C Humidity={1:0.1f}%'.format(t, h)

        url = "http://IoT-Cloud Hub IP address:48080/api/v1/event"

        payload = {"origin": 1471806386919, "device": "Device1",
                   "readings": [{"origin": 1471806386919, "name": "device", "value": 'Device1'},
                                {"origin": 1471806386919, "name": "time", "value": date},
                                {"origin": 1471806386919, "name": "temperature", "value": t},
                                {"origin": 1471806386919, "name": "humidity", "value": h}]}

        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        response = requests.post(url, data=json.dumps(payload), headers=headers)

        print(response.text)
        print(response.headers)

        time.sleep(10)  ##delay time 100 seconds

        url = "http://IoT-Cloud Hub IP address:5000" ##flask에 api 요청
        response = requests.get(url, headers=headers)
