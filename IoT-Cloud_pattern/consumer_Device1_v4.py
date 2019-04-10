import threading, logging, time
import multiprocessing
import msgpack
from kafka import KafkaProducer
from kafka import TopicPartition
from kafka import KafkaConsumer
from kafka.errors import KafkaError

import requests, json
import time
import os
import sys
import subprocess
import urllib, urllib2

from time import localtime, strftime
from pyowm import OWM




cmd ="curl -XPOST 'influxdb:8086/query' --data-urlencode 'q=CREATE DATABASE 'Sensordata''"
subprocess.call([cmd], shell=True)

timeout = 100
actual_data=[]

consumer = KafkaConsumer('Device1', bootstrap_servers=['kafka-broker-1:9092'])
partitions = consumer.poll(timeout)
while partitions == None or len(partitions) == 0:

        consumer = KafkaConsumer('Device1', bootstrap_servers=['kafka-broker-1:9092'])
        message = next(consumer)
        #print(message.value)

        str1 = message.value
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

        time_val2 = time_val.partition(':')
        time_val = time_val2[2]

        temperature2 = temperature.split(':')
        temperature = temperature2[1]

        humidity2 = humidity.split(':')
        humidity = humidity2[1]

        #variables = "dev1"
        Aircon_temp = '18'

        API_key = '5ae0b0ddfba53bb1acc0ec416e931177'  ## information for weather configure
        owm = OWM(API_key)
        obs = owm.weather_at_place('Gwangju')
        obs = owm.weather_at_coords(35.15972, 126.85306)

        w = obs.get_weather()
        weather_stat = w.get_status()
        ext_temperature = w.get_temperature(unit='celsius')['temp']
        ext_temperature = "{0:0.2f}".format(ext_temperature)
        ext_humidity = w.get_humidity()
        ext_humidity = "{0:0.2f}".format(ext_humidity)

        if weather_stat=='Thunderstorm':
                weather_stat = 1
        if weather_stat=='Drizzle':
                weather_stat = 2
        if weather_stat=='Rain':
                weather_stat = 3
        if weather_stat=='Snow':
                weather_stat = 4
        if weather_stat=='Atmosphere':
                weather_stat = 5
        if weather_stat=='Clear':
                weather_stat = 6
        if weather_stat=='Clouds':
                weather_stat = 7
        if weather_stat=='Mist':
                weather_stat = 8
        if weather_stat=='Haze':
                weather_stat = 9

        print("%s,%s,%s,%s,%s,%s") % (temperature, humidity, Aircon_temp, ext_temperature, ext_humidity, weather_stat)
        cmd = "curl -XPOST 'influxdb:8086/write?db=Sensordata' --data-binary 'dev1,location=Gwangju temperature=%s,humidity=%s,Aircon_temp=%s,ext_temperature=%s,ext_humidity=%s,weather_stat=%s'" % (temperature, humidity, Aircon_temp, ext_temperature, ext_humidity, weather_stat)
        subprocess.call([cmd], shell=True)


        #variables = "dev1"
        #cmd = "curl -i -XPOST 'http://localhost:8086/write?db=Sensordata' --data-binary '%s,host=%s,region=Gwangju_tower_room1 value=%s\n%s,host=Aircon_temp,region=Gwangju_tower_room1 value=%s'" % (variables, device_name, humidity, variables, Aircon_temp)
        #subprocess.call([cmd], shell=True)

        #variables = "Aircon_temp"
        #cmd = "curl -i -XPOST 'http://localhost:8086/write?db=Sensordata' --data-binary 'Aircon_temp,region=Gwangju %s=\"%s\"'" % (variables, weather_stat)
