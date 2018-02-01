// assign pin num
int right_pin = 5;
int left_pin = 6;
int forward_pin = 4;
int reverse_pin = 3;

// duration for output
int time = 200;
// initial command
int command = 0;

void setup() {
  pinMode(right_pin, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(11, OUTPUT);
 
  pinMode(left_pin, OUTPUT);
  pinMode(forward_pin, OUTPUT);
  pinMode(reverse_pin, OUTPUT);
  //digitalWrite(5, 130);
  //analogWrite(5, 130);
  //digitalWrite(11, HIGH);
  //analogWrite(11, 255);
  Serial.begin(115200);
}

void loop() {
  //receive command
  //analogWrite(5, 150); // Send PWM signal to L298N Enable pin
  //analogWrite(11, 200);
  digitalWrite(5,HIGH);
  digitalWrite(11,HIGH);
  if (Serial.available() > 0){
    command = Serial.read();
    Serial.println(command);
    send_command(command,time);
  }
  else{
    reset();
  }
 // right(500);
   //forward_right(50);
  
  
//  Serial.println(pwmOutput);
  
    //reset();
    //delay(1000);
   /*forward(500);
   reverse(500);
   right(500);
   left(500);
   reset();*/
   /*forward_right(1000);
   reset();
   delay(1000);
   reverse_right(1000);*/
   
   }

void right(int time)
{
  
  digitalWrite(right_pin, HIGH);
  digitalWrite(left_pin, LOW);
  digitalWrite(forward_pin, HIGH);
  digitalWrite(reverse_pin, LOW);
  delay(time);
}

void left(int time){
  digitalWrite(right_pin, LOW);
  digitalWrite(left_pin, HIGH);
  digitalWrite(forward_pin, HIGH);
  digitalWrite(reverse_pin, LOW);
  delay(time);
}

void forward(int time)
{
  
  digitalWrite(forward_pin, HIGH);
  digitalWrite(reverse_pin, LOW);
  digitalWrite(right_pin, LOW);
  digitalWrite(left_pin, LOW);
  delay(time);
}

void reverse(int time){
  digitalWrite(forward_pin, LOW);
  digitalWrite(reverse_pin, HIGH);
  delay(time);
}

void forward_right(int time){
  digitalWrite(forward_pin, HIGH);
  digitalWrite(right_pin, HIGH);
  delay(time);
}

void reverse_right(int time){
  digitalWrite(reverse_pin, HIGH);
  digitalWrite(right_pin, HIGH);
  delay(time);
}

void forward_left(int time){
  digitalWrite(forward_pin, HIGH);
  digitalWrite(left_pin, HIGH);
  delay(time);
}

void reverse_left(int time){
  digitalWrite(reverse_pin, HIGH);
  digitalWrite(left_pin, HIGH);
  delay(time);
}

void reset(){
  digitalWrite(right_pin, LOW);
  digitalWrite(left_pin, LOW);
  digitalWrite(forward_pin, LOW);
  digitalWrite(reverse_pin, LOW);
}

void send_command(char command, int time)
{
  switch (command)
  {

     //reset command
     Serial.println(command);
     case 0: reset(); break;

     // single command
     case 1: forward(time); break;
     case 2: reverse(time); break;
     case 3: right(time); break;
     case 4: left(time); break;

     //combination command
     case 6: right(time); break;
     case 7: left(time); break;
     case 8: reverse_right(time); break;
     case 9: reverse_left(time); break;

     default: forward(time);
     
    }
}
