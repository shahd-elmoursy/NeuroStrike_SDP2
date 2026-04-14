import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc, properties=None):
   global flag_connected
   flag_connected = 1
   client_subscriptions(client)
   print("Connected to MQTT server")

def on_disconnect(client, userdata, flags, rc, properties=None):
   global flag_connected
   flag_connected = 0
   print("Disconnected from MQTT server")
   
# a callback functions 
def callback_esp32(client, userdata, msg):
    print(msg.topic, ' - Data: ', msg.payload.decode('utf-8'))

def callback_rpi_broadcast(client, userdata, msg):
    print('RPi Broadcast message:  ', str(msg.payload.decode('utf-8')))

def client_subscriptions(client):
    client.subscribe("traffic/#")
    client.subscribe("hazards/#")
    client.subscribe("environment/#")
    client.subscribe("rpi1/broadcast")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "rpi_client1", clean_session=True) #this should be a unique name
flag_connected = 0

client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.message_callback_add('traffic/ultrasonic_1', callback_esp32)
client.message_callback_add('traffic/motion_1', callback_esp32)
client.message_callback_add('hazards/water_level_1', callback_esp32)
client.message_callback_add('hazards/flame_alert_1', callback_esp32)
client.message_callback_add('environment/temperature_1', callback_esp32)
client.message_callback_add('environment/light_1', callback_esp32)
client.message_callback_add('rpi1/broadcast', callback_rpi_broadcast)
client.connect('192.168.0.134',1883)
# start a new thread
client.loop_start()
print("......client setup complete............")


while True:
    time.sleep(4)
    if (flag_connected != 1):
        print("trying to connect MQTT server..")
        
