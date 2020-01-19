# McWics-2020
Dance Dance Revolution Mini :)

</br>
The goal of our project is to have a Hack McWicks version of Just Dance with the dance moves being recognized through wrnch's SDK via a live webcam. If the dance move is correct, the LED strip outputs a green color, otherwise a red color.

</br>

We used an Arduino to power the LED strip and the python library pyserial to send commands from the python script to the arduino to trigger the correct color sequences for the LED strip.

</br>

The live camera feed gets processed through wrnch's SDK [which we obtained as part of the hackathon event], compares it to the data of the correct move (from a picture we took and processed). Using vector calculations we can get an estimate of the resemblance of the two dance poses (live webcam feed and saved image). A number between 0 (high similarity) and 2 (low similarity) gets outputed and updated multiple times per second. If this number gets below a certain threshold, the python script sends the command to the arduino to trigger color sequence : orange if below than 1 and green if below than 0.8.  Otherwise, red flashes.

</br>


To connect Arduino and python code via pyserial:

- Install pyserial
  - pip install pyserial

- Configure pyserial
  - go in python file
    - insert the following code at the beginning, COMx being the serial port connecting your computer to Arduino
      - Import serial
      - ser1 = serial.Serial('COM1', 9600)
    - insert the following code to send command to Arduino, replace 'a' by any single character you want
      - ser1.write(’a’.encode())
