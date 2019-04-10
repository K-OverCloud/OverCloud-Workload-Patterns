#!/bin/bash


kubectl apply -f zookeeper.yaml
kubectl apply -f kafka.yaml
kubectl apply -f influx_chro.yaml
kubectl apply -f api_server.yaml
kubectl apply -f consumer_device1.yaml
kubectl apply -f EdgeX2.yaml



