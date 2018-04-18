close all
I=imread('rice.png');

R=1;
L1=LBP(I,R);
figure,
hist((L1(:)),16)

R=1;
L1=LBP_u2(I,R);
figure,
hist((L1(:)),10)