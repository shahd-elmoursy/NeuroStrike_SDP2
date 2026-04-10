#include <WiFi.h>
#include <PubSubClient.h>

// Replace the SSID/Password details as per your wifi router
const char* ssid = "yourSSID";
const char* password = "yourPassword";

// Replace your MQTT Broker IP address here:
const char* mqtt_server = "192.168.0.133";


WiFiClient espClient;
PubSubClient client(espClient);

long lastMsg = 0;

// Ultrasonic Sensor Pins
const int trigPin = 5;
const int echoPin = 18;

// Motion Sensor Pins
const int motionPin = 4;
int motionState = 0;

#define ledPin 2
#define SOUND_SPEED 0.034
#define CM_TO_INCH 0.393701

long duration;
float distanceCm;

void blink_led(unsigned int times, unsigned int duration){
  for (int i = 0; i < times; i++) {
    digitalWrite(ledPin, HIGH);
    delay(duration);
    digitalWrite(ledPin, LOW); 
    delay(200);
  }
}

void setup_wifi() {
  delay(50);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  int c=0;
  while (WiFi.status() != WL_CONNECTED) {
    blink_led(2,200); //blink LED twice (for 200ms ON time) to indicate that wifi not connected
    delay(1000); //
    Serial.print(".");
    c=c+1;
    if(c>10){
        ESP.restart(); //restart ESP after 10 seconds
    }
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  
}

void connect_mqttServer() {
  // Loop until we're reconnected
  while (!client.connected()) {

        //first check if connected to wifi
        if(WiFi.status() != WL_CONNECTED){
          //if not connected, then first connect to wifi
          setup_wifi();
        }

        //now attemt to connect to MQTT server
        Serial.print("Attempting MQTT connection...");
        // Attempt to connect
        if (client.connect("ESP32_client1")) { // Change the name of client here if multiple ESP32 are connected
          //attempt successful
          Serial.println("connected");
          // Subscribe to topics here
          client.subscribe("rpi_1/broadcast");
          //client.subscribe("rpi/xyz"); //subscribe more topics here
          
        } 
        else {
          //attempt not successful
          Serial.print("failed, rc=");
          Serial.print(client.state());
          Serial.println(" trying again in 2 seconds");
    
          blink_led(3,200); //blink LED three times (200ms on duration) to show that MQTT server connection attempt failed
          // Wait 2 seconds before retrying
          delay(2000);
        }
  }
  
}

//this function will be executed whenever there is data available on subscribed topics
void callback(char* topic, byte* message, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  String messageTemp;
  
  for (int i = 0; i < length; i++) {
    Serial.print((char)message[i]);
    messageTemp += (char)message[i];
  }
  Serial.println();

  // Check if a message is received on the topic "rpi/broadcast"
  if (String(topic) == "rpi_1/broadcast") {
      if(messageTemp == "15"){
        Serial.println("Action: blink LED");
        blink_led(1,1250); //blink LED once (for 1250ms ON time)
      }
  }

  //Similarly add more if statements to check for other subscribed topics 
}

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(115200);
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin, INPUT); // Sets the echoPin as an Input
  
  pinMode(motionPin, INPUT); // Sets the motionPin as an Input

  setup_wifi();
  client.setServer(mqtt_server,1883);//1883 is the default port for MQTT server
  client.setCallback(callback);
}

void loop() {
  
  if (!client.connected()) {
    connect_mqttServer();
  }

  client.loop();
  
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);
  
  // Calculate the distance
  distanceCm = duration * SOUND_SPEED/2;

  char buffer[20]; // Enough space for the number as text

  // Convert float to string with 3 decimal places
  snprintf(buffer, sizeof(buffer), "%.3f", distanceCm);

  long now = millis();
  if (now - lastMsg > 4000) {
    lastMsg = now;
    
    Serial.print("Distance (cm): ");
    Serial.println(distanceCm);
    client.publish("traffic/ultrasonic_1", buffer);

    memset(buffer, '\0', sizeof(buffer));
    
    delay(1000);
  }


  // Motion Sensor Code: 
  static int lastMotionState = LOW;
  motionState = digitalRead(motionPin);

  if (motionState != lastMotionState) {
  lastMotionState = motionState;

    if (motionState == HIGH) {
      Serial.println("Motion Detected!");
      client.publish("traffic/motion_1", "Motion Detected!");
    } else {
      Serial.println("No Motion");
      client.publish("traffic/motion_1", "No Motion");
    }
  }
  
}
