clear
%%
% sample parameters
Fs = 8192; % sample frequency
Ts = 1/Fs; % sample period
N = 100000; % number of sample nodes
t = Ts:Ts:N*Ts; % time sequence
f = (0:N/2-1)*Fs/N; % frequency sequence

%%
%{
	gearbox parameters -- gear ring fixed
	assume sun gear shaft as input, carrier shaft as out put
%}
Zp  = 36; Zs = 28; Zr = 100; % number of teeth of planet gear, sun gear and gear ring
							% must satisfy Zs + 2*Zp = Zr 			
np  = 4; % number of planet gears
tr  = (Zs+Zr)/Zs; % transfer ratio of gearbox(assume carrier to be fixed, the rotation
				  % frequency of gear ring can be rotation frequency of carrier, and
				  % rotation frequency of sun gear should minus frequency of carrier)
is_steady = 0;
is_vary = 1-is_steady;
if is_steady
	% for steady condition
	frs = 25; % rotation frequency of sun gear
else	
	% for varying condition
	frs = 0.5*t;
end
kcon = 0.5; % coefficient used to time the 'frs' term
fcon = 0; % frequency that describe how condition varies


frc = frs/tr; % rotational frequency of carrier
fm  = frc*Zr; % mesh frequency
frp_rel = fm/Zp; % the relative(to gear ring) rotation frequency of planet gear
frp_abs = frp_rel + frc; % the absolute rotation frequency of planet gear


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
	2012  Feng Zhipeng model
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%{
	vibration model for distributed fault induced gearbox(normal one)
%}
% characteristic frequency
fs = fm/Zs; fp = fm/Zp; fr = fm/Zr;
% condition parameters
Acon = kcon*frs;
Mcon = is_steady+(1-is_steady)*Acon.*cos(2*pi*fcon.*t); % the varying condition modulation
% initial parameters
phi_s=0; phi_p=0; phi_r=0; PHI_s=0; PHI_p=0; PHI_r=0; theta=pi/2; % 初始相位
As=1; Ap=1; Ar=1; Bs=1; Bp=1; Br=1; % 误差或分布式故障引起的条幅和调幅强度
K = 0.5; % overall scalar
% vibration model
x_d = K*Mcon.*(1-cos(2*pi*np*t)).*(1+As*cos(2*pi*fs.*t+PHI_s)...
		+Ap*cos(2*pi*fp.*t+PHI_p)+Ar*cos(2*pi*fr.*t+PHI_r)).*...
		cos(2*pi*fm.*t+Bs*sin(2*pi*fs.*t+phi_s)+Bp*sin(2*pi*fp.*t+phi_p)...
		+Br*sin(2*pi*fr.*t+phi_r)+theta);
% x_d_n = awgn(x_d,1); % add Gaussian white noise with 1 dB
y_d = abs(fft(x_d)); y_d = y_d(1:N/2); % frequency spectrum

%%
%{
	vibration model for local fault induced gearbox
%}
% characteristic frequency
fp = fm/Zp; fs = fm/Zs*np; fr = fm/Zr*np;
% initial parameters
A = 1; B = 0.5; phi1 = 0; phi2 = 0; phi3 = 0;
%%%%%%%%%%%%%[planet gear local fault]%%%%%%%%%%%%%%%%%
% vibration model	
x_p = Mcon.*(1-cos(2*pi*frc.*t)).*(1+A*cos(2*pi*fp.*t+phi1))...
		.*cos(2*pi*fm.*t+B*sin(2*pi*fp.*t+phi2)+phi3)+x_d;
% x_p = awgn(x_p, 2); % add Gaussian white noise with 1 dB
y_p = abs(fft(x_p)); y_p = y_p(1:N/2); % frequency spectrum

%%%%%%%%%%%%%[sun gear local fault]%%%%%%%%%%%%%%%%%
% vibration model	
x_s = Mcon.*(1-cos(2*pi*frs.*t)).*(1+A*cos(2*pi*fs.*t+phi1))...
		.*cos(2*pi*fm.*t+B*sin(2*pi*fs.*t+phi2)+phi3)+x_d;
x_s = awgn(x_s, 2); % add Gaussian white noise with 1 dB
y_s = abs(fft(x_s)); y_s = y_s(1:N/2); % frequency spectrum

%%%%%%%%%%%%%[sun gear local fault]%%%%%%%%%%%%%%%%%
% vibration model	
x_r = Mcon.*(1+A*cos(2*pi*fr.*t+phi1))...
		.*cos(2*pi*fm.*t+B*sin(2*pi*fr.*t+phi2)+phi3)+x_d;
x_r = awgn(x_r, 2); % add Gaussian white noise with 1 dB
y_r = abs(fft(x_r)); y_r = y_r(1:N/2); % frequency spectrum

figure
subplot(211)
plot(t,x_p)
subplot(212)
plot(f,y_p)
