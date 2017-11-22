% this program do the t-f trans for the data and turn them in images
% batchly

%% explain the filename to the machine
%fileset = ['20150407pmt_12k_50.txt';'20150407na_12k_50.txt';'20150408psf_12k_50.txt'];
filename = '20150407pmt_12k_50.txt';
file = load(filename);
N = 110000; % wanted sample points
% 
S = regexp(filename,'_','split'); % split the string
Fs = char(S(2));
Fs = str2double(Fs(1:2)) * 1000; % sample frequency
S = regexp(char(S(3)),'\.','split'); 
rf = str2double(char(S(1))); % 
Ts = 1/Fs;

x0 = file(1:N,4); % data for use
clear file
x0 = x0';
t = 0:Ts:(N-1)/Fs; % time series

switch(filename(9:10))
    case 'na'
        dataname = 'normal';
    case 'pm'
        dataname = 'pgmt';
    case 'ps'
        dataname = 'pgsw';
    otherwise
        dataname = 'unknown';
end

%% divide data
%
lengthOfEachSample=2400;
divideGap=20;
numOfPieces=5000;
targetType=1;
[originSet,~] = divideData(x0,lengthOfEachSample,divideGap,numOfPieces,targetType); 
[timeSet,~] = divideData(t,lengthOfEachSample,divideGap,numOfPieces,targetType);

%% cwt
%{
mscale=200;
scale = 1:mscale;
wavename='cmor3-3';
wcf=centfrq(wavename); 
cw = cwt(x0,scale,wavename);
x=abs(cw);
% normalization
xmax = max(max(x));
xmin = min(min(x));
xgap = xmax-xmin;
[a,b] = size(x);
ZZ = zeros(a,b);
for ii = 1:a
    for jj = 1:b
        ZZ(ii,jj) = (x(ii,jj)-xmin)/xgap;
    end
end
figure
mesh(x)
view(0,90)
set(gca,'ylim',[0,mscale], 'xlim', [0,lengthOfEachSample])
figname = [dataname ',CWT' ',rspeed-' num2str(rf) ',sfre-' num2str(Fs)];
title(figname)
xlabel('time [s]'),ylabel('frequency [Hz]')
%}
%% make the image set
tic
mscale = 200;
scale = 1:mscale;
wavename='cmor3-3';
for ii = 1:numOfPieces
    x = abs(cwt(originSet(:,ii),scale,wavename));
    % desample in the time scale, 'max-pool'-[1,10]
    d_scale = 10;
    desampled_length = lengthOfEachSample/d_scale;
    temp = zeros(mscale, desampled_length);
    for jj = 1:desampled_length
        temp(:,jj) = max(x(:,jj:jj+d_scale-1),[],2); % cal the max in the 2nd dim
    end
    x = temp; 
    xmax = max(max(x));
    xmin = min(min(x));
    x = uint8((x-xmin)/(xmax-xmin)*256);
    savepath = '/home/codeplay2017/code/lab/code/paper/realwork/image/trainset3/';
    figname = [dataname ',CWT' ',rspeed-' num2str(rf) ',sfre-' num2str(Fs)];
    imgname = [figname '_' num2str(ii) '.png'];
    imwrite(x,[savepath imgname],'png')
end
toc
%}


