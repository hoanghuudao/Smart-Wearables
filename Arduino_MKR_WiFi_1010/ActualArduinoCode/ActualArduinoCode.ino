#include <WiFiNINA.h>

// WiFi credentials
const char* ssid = "kumibotte";
const char* password = "kumibotte";

// Server credentials
const IPAddress serverIP(192, 168, 167, 225);
const int serverPort = 8080;

// Bend sensor pins
const int sensorPin1 = A1;
const int sensorPin2 = A2;
const int sensorPin3 = A3;
const int sensorPin4 = A4;

WiFiClient client;

void setup() {
  // Serial.begin(115200);
  // while (!Serial); // Wait for Serial Monitor to open (only needed for debugging)

  if (WiFi.status() == WL_NO_MODULE) {
    //Serial.println("WiFi module not detected!");
    while (true); // Halt execution
  }

  initWiFi();
  connectToServer();
}

void initWiFi() {
  //Serial.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  unsigned long startAttemptTime = millis();
  
  while (WiFi.status() != WL_CONNECTED) {
    //Serial.print('.');
    delay(1000);
    
    if (millis() - startAttemptTime >= 15000) {
      //Serial.println("\nFailed to connect to WiFi. Restarting...");
      WiFi.disconnect();
      WiFi.begin(ssid, password);
      startAttemptTime = millis();
    }
  }

  //Serial.println("\nConnected to WiFi!");
  //Serial.print("IP Address: ");
  //Serial.println(WiFi.localIP());
}

void connectToServer() {
  //Serial.println("Connecting to server...");

  while (!client.connect(serverIP, serverPort)) {
    //Serial.println("Failed to connect to server. Retrying...");
    delay(3000);
  }

  //Serial.println("Connected to server!");
}

void loop() {
  if (client.connected()) {
    int sensorValue1 = analogRead(sensorPin1);
    int sensorValue2 = analogRead(sensorPin2);
    int sensorValue3 = analogRead(sensorPin3);
    int sensorValue4 = analogRead(sensorPin4);

    String data = String(sensorValue1) + "," + 
                  String(sensorValue2) + "," + String(sensorValue3) + "," + 
                  String(sensorValue4);

    client.println(data);
    //Serial.print("Sent: ");
    //Serial.println(data);

    delay(50);
    } else {
    //Serial.println("Server disconnected. Reconnecting...");
    connectToServer();
  }
}
