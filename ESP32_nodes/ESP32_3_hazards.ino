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

// Flame Detection Sensor Pins
const int flamePin = 34; //Analog pin
int flameValue = 0;
int flameThreshold = 1500;

// Water level Sensor Pins
const int trigPin = 5;
const int echoPin = 18;


#define ledPin 2
#define SOUND_SPEED 0.034

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
        if (client.connect("ESP32_client3")) { // Change the name of client here if multiple ESP32 are connected
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
  
  // Flame Setup
  pinMode(flamePin, INPUT); 
  
  // Water Level Setup
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  analogSetAttenuation(ADC_11db);  // Better ADC range
  
  setup_wifi();
  client.setServer(mqtt_server,1883);//1883 is the default port for MQTT server
  client.setCallback(callback);
}

void loop() {
  
  if (!client.connected()) {
    connect_mqttServer();
  }

  client.loop();

  long now = millis();
  if (now - lastMsg > 4000) {
    lastMsg = now;
    
    // Flame Detection
    flameValue = analogRead(flamePin);

    char flameBuffer[20];
    snprintf(flameBuffer, sizeof(flameBuffer), "%d", flameValue);

    Serial.print("Flame Value: ");
    Serial.println(flameValue);

    if (flameValue > flameThreshold) {
      Serial.println("Flame Detected!");
      client.publish("hazards/flame_alert_1", "Flame Detected!");
      blink_led(1,500);
    }
    else {
      Serial.println("No Flame!");
      client.publish("hazards/flame_alert_1", "No Flame!");
    }
    
    // Water Level

    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    digitalWrite(trigPin, HIGH);
    delayMicroseconds(20);
    digitalWrite(trigPin, LOW);

    duration = pulseIn(echoPin, HIGH, 60000);
    /// Section added
    if (duration == 0) {
      Serial.println("No echo received!");
      client.publish("hazards/water_level_1", "No Echo");
      return;  // Skip this cycle safely
    }

    distanceCm = duration * SOUND_SPEED / 2;

    char waterBuffer[20];
    snprintf(waterBuffer, sizeof(waterBuffer), "%.4f", distanceCm);
    // Section added
    Serial.print("Duration: ");
    Serial.println(duration);

    Serial.print("Water Distance (cm): ");
    Serial.println(distanceCm);

    client.publish("hazards/water_level_1", waterBuffer);
  }
}
