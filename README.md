# Smart Energy service

Smart Energy service is one of the IoT-Cloud service examples based on kubernetes to provide energy saving connected with IoT devices, gateway and cloud.

## Requirement

* IoT-Cloud Hub (Speacially prepared hardware for IoT Gateway)
* Ubuntu 16.04
* Kubernetes

## Deployment application functionalities

In this step, you should run following commands on IoT-Cloud Hub.

Before you deploy Smart Energy application functionalities based on kubernetes on your IoT-Cloud Hub you need to move your directory.

$ cd SmartX-MicroBox/application_functionality/Smart-Energy-service/Smart-Energy-service-yaml

To deploy Smart Energy application functionalities

$ ./Smart_Energy.sh

## IoT Devices functionalities

In this step, you need to run following commands on IoT devices.

Before you execute IoT devices functionalities you need to move your directory as follows.

$ cd /SmartX-MicroBox/application_functionality/Smart-Energy-service$ cd Device_functions/

$ ./TempHumsend.py


