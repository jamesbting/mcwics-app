#include <Adafruit_NeoPixel.h>

#define PIN 6

// Parameter 1 = number of pixels in strip
// Parameter 2 = pin number (most are valid)
// Parameter 3 = pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
Adafruit_NeoPixel strip = Adafruit_NeoPixel(60, PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  strip.begin();
  strip.show(); // Initialize all pixels to 'off'
  Serial.begin(9600);
}


// Waits for person to arrive in LED strip circle
void loop() {
  if(Serial.available() > 0)
  {
    if(Serial.read() == '0')
    {
      detected();  // Once person arrive in LED strip circle
    }
  }  

    else
    {
      waitingColor(2);  // Rainbow Color wheel while wainting for player to arrive
    }
}



// Waiting for player to be ready (5secs) + Ready Set Go 
void detected(){
  for(uint16_t i=0; i<3; i++) {
    //glow(strip.Color(255, 255, 255),2);  // White glow up and down i times
  }

  glow(strip.Color(0, 255, 0),0.5);  // Red glow up and down -- ready
  glow(strip.Color(100, 255, 0),0.5);  // Orange glow up and down -- set
  glow(strip.Color(255, 0, 0),0.5);  // Green glow up and down -- go

  danceDetection();  // Start of the game + detection of movements with adjusting color

}


// Rainbow Color wheel
void waitingColor(uint8_t wait) {
  uint16_t i, j;
  for(j=0; j<256; j++) {
    for(i=0; i<strip.numPixels(); i++) {
      strip.setPixelColor(i, Wheel((i+j) & 255));
    }
    strip.show();
    delay(wait);
  }
}


// Fast Rainbow Color wheel
void gameColor(uint8_t wait) {
  uint16_t i, j;
  for(j=0; j<256; j++) {
    for(i=0; i<strip.numPixels(); i++) {
      strip.setPixelColor(i, Wheel((i+j) & 255));
      strip.setBrightness(j);
    }
    strip.show();
    delay(wait);
  }
}



// Start of the game + detection of movements with adjusting color
void danceDetection(){
  for(uint16_t j=0; j<5; j++) {
    if (j==4){
      j = 0;
    }
      if(Serial.available() > 0)
  {
    uint16_t input = Serial.read();
    if(input == '1')
    {
      glow(strip.Color(0, 255, 0),0.5);  // Red glow up and down -- wrong
    }  

    else if(input == '2')
    {
      glow(strip.Color(100, 255, 0),0.5);  // Orange glow up and down -- almost
    }

    else if(input == '3')
    {
      colorWipe(strip.Color(255, 0, 0), 40);  // Green wipe -- yes
      }

    else if(input == '4')
    {
      endGame(strip.Color(127, 127, 127), 50, 30);  // Green wipe -- yes
      return;
      
      
      }

  }
    else
    {
      gameColor(0.5);  // Fast rainbow Color wheel between moves
    }
    }
}


  
// Glows one color up and down
void glow(uint32_t c, uint8_t wait) {
  uint16_t i, j;
  for(j=0; j<256; j++) {
    for(i=0; i<strip.numPixels(); i++) {
      strip.setPixelColor(i,c);
      strip.setBrightness(j);
    }
    strip.show();
    delay(wait);
  
  }
  for(j=254; j>1; j--) {
    for(i=0; i<strip.numPixels(); i++) {
      strip.setPixelColor(i,c);
      strip.setBrightness(j);
    }
    strip.show();
    delay(wait);
  
  }
  strip.setBrightness(255);
}






//// Fill the dots one after the other with a color
void colorWipe(uint32_t c, uint8_t wait) {
  for(uint16_t i=0; i<strip.numPixels(); i++) {
      strip.setPixelColor(i, c);
      strip.show();
      delay(wait);
  }
}



//// Input a value 0 to 255 to get a color value.
//// The colours are a transition r - g - b - back to r.
uint32_t Wheel(byte WheelPos) {
  if(WheelPos < 85) {
   return strip.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
  } else {
   WheelPos -= 170;
   return strip.Color(0, WheelPos * 3, 255 - WheelPos * 3);
  }
}

////Theatre-style crawling lights
void endGame(uint32_t c, uint8_t wait, uint32_t cycle) {
  for (int j=0; j<cycle; j++) {  //do 'cycle' cycles of chasing
    for (int q=0; q < 3; q++) {
      for (int i=0; i < strip.numPixels(); i=i+3) {
        strip.setPixelColor(i+q, c);    //turn every third pixel on
      }
      strip.show();

      delay(wait);

      for (int i=0; i < strip.numPixels(); i=i+3) {
        strip.setPixelColor(i+q, 0);        //turn every third pixel off
      }
    }
  }
}
