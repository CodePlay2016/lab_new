% 试验台： 行星齿轮箱+二级普通齿轮箱（实验齿轮箱）
% planet gear teeth: 36
% sun gear teeth 28
% ring gear teeth: 100
% num of planet gears: 4
clc;
clear;
MSpd=300:300:1500;   %电机转速，rpm(if1500)
MSpd=MSpd';
PGInf=MSpd/60;       %太阳轮转轴频率，Hz(25)
TGInf=PGInf*28/128;  %实验齿轮箱输入转轴频率，即行星架输出转轴频率，Hz(5.47)
                     %128 = 28 + 100, 28/128 is the transfer ratio
PGMeshf=TGInf*128;   %啮合频率，fm = (fsun-fcarry)*Zsun=fcarry*Zring=fplane_self*Zplanet, Hz(700)
TGMeshf1=TGInf*100;  %实验齿轮箱一级齿轮对啮合频率，Hz(546.88)
TGShf=TGMeshf1/29;   %实验齿轮轴频率，Hz(18.86)
TGMeshf2=TGShf*36;   %实验齿轮箱二级齿轮对啮合频率，Hz(678.88)
TGOutf=TGMeshf2/90;  %实验齿轮箱输出轴频率，Hz(7.543)
PGpassf=TGInf*4;     %行星轮通过频率，行星架的转频×行星轮个数(21.88)

% fs=12000;   
% N=8192;    
% n=1:N;
% t=n/fs;
% fx=13;
% fy=230;
% x=cos(2*pi*fx*t);
% y=10*cos(2*pi*fy*t);
% z=x+y;
% data=z;
% ftT = 1/fs;                    
% ftf = fs/2*linspace(0,1,ftNFFT/2+1);
% ftY = fft(data,ftNFFT);%/ftNFFT;
% subplot(311);
% plot(ftf,2*abs(ftY(1:ftNFFT/2+1)),'b');
% ylabel('Vibration Amplitude');xlabel('Frequency (Hz)');title('Test Gearbox X Vibraiton FFT Spectrum');
