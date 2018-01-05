% this program do the t-f trans for the data and turn them in images
% batchly
clear
inputpath = '/home/codeplay2017/code/lab/code/paper/realwork/data/12k/varying_condition_test/';
savepath = '/home/codeplay2017/code/lab/code/paper/realwork/image/wen_data/fft_divided/fft_series_step1_2048_2speeds/';
filelist = dir(inputpath);
file_num = length(filelist);

for ff = 3:file_num
%% explain the filename to the machine
filename = filelist(ff).name;
% filename = '20150407na_12k_30-0.txt';
disp(['dealing with ' filename])

file = load([inputpath filename]);
% N = 160000; % wanted sample points

% 
S = regexp(filename,'_','split'); % split the string
Fs = char(S(2));
Fs = str2double(Fs(1:2)) * 1000; % sample frequency

S = regexp(char(S(3)),'\.','split'); 
rf = char(S(1)); %
Ts = 1/Fs;

if strcmp(rf, '30-0')
    x0 = file(40001:105000,4); % data for pmt30-0/normal30-0/psf30-0
elseif strcmp(rf, '0-50')
    if filename(9:10) == 'na'
        x0 = file(65001:130000,4); % data for normal0-50/
    elseif filename(9:10) == 'pm'
        x0 = file(55001:120000,4); % data for pmt0-50/
    elseif filename(9:10) == 'ps'
        x0 = file(85001:150000,4); % data for psw0-50
    end
else
    x0 = file(:,4);
end
N = length(x0)
t = 0:Ts:(N-1)*Ts;
fs = 12000;
f = (0:N/2-1)*fs/N;
    
% 
figure(1)
plot(x0)
title('original signal')
% figure(2)
% yf = abs(fft(x0));
% plot(f, yf(1:N/2)*2/N)
% title('frequency spectrum')
clear file
x0 = x0';
% t = 0:Ts:(N-1)/Fs; % time series

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

% if ~strcmp(dataname, 'pgmt')
%     continue
% end


% ro_speed = str2double(rf)*pi*ones(1, N);
% w = zeros(1,N);
% w(1) = ro_speed(1) * Ts;
% for ii = 2:N
%     w(ii) = w(ii-1) + ro_speed(ii)*Ts;
% end
% dwr = 10*pi*Ts;
% wr = w(1):dwr:(N-1)*dwr;
% resampled = spline(w, x0, wr);

% N = 40000;

% figure
% subplot(311)
% f = (0:N/2-1)*fs/N;
% yf = abs(fft(x0(1:N)));
% plot(f, yf(1:N/2)*2/N)

% subplot(312)
% f_resampled = abs(fft(resampled(1:N)));
% wf = (0:N/2-1)*(1/dwr)/N;
% plot(wf, f_resampled(1:N/2)*2/N)

% subplot(313)
% hx = hilbert(resampled(1:N));
% hx = abs(hx);
% % hx = sqrt(resampled(1:N).^2+imag(hx).^2);
% f_hx = abs(fft(hx));
% plot(wf, f_hx(2:N/2+1)*2/N)
% suptitle([dataname, ' ', rf])
% % print(gca,'-dpng',['figure/' filename '.png'])
% N = length(x0);


%% divide data
%
lengthOfEachSample=4096;
divideGap=1;
numOfPieces = 60000;
targetType=1;
% [originSet,~] = divideData(resampled,lengthOfEachSample,divideGap,numOfPieces,targetType); 
[originSet,~] = divideData(x0,lengthOfEachSample,divideGap,numOfPieces,targetType); 
% [timeSet,~] = divideData(t,lengthOfEachSample,divideGap,numOfPieces,targetType);
%}

%% fft
%
originSet = abs(fft(originSet));
originSet = originSet(1:2048,:);
%}

%% cwt
%{
mscale=200;
scale = 1:mscale;
wavename='cmor3-3';
wcf=centfrq(wavename);
x0 = abs(hilbert(x0));
cw = cwt(x0,scale,wavename);
x=abs(cw);
% normalization
% xmax = max(max(x));
% xmin = min(min(x));
% xgap = xmax-xmin;
% [a,b] = size(x);
figure
mesh(x)
view(0,90)
% set(gca,'ylim',[0,mscale], 'xlim', [0,lengthOfEachSample])
figname = [dataname ',CWT' ',rspeed-' num2str(rf) ',sfre-' num2str(Fs)];
title(figname)
xlabel('time [s]'),ylabel('frequency [Hz]')
%}
%% make the image set by cwt
%{
mscale = 256;
scale = 1:mscale;
wavename='cmor3-3';
result = zeros(lengthOfEachSample, numOfPieces);
for ii = 1:numOfPieces
    d_scale = 8;
    desampled_length = lengthOfEachSample/d_scale;
    % 1. cwt the data
    x = abs(cwt(originSet(:,ii),scale,wavename));
    % desample in the time scale, 'max-pool'-[1,10]
    temp = zeros(mscale, desampled_length);
    for jj = 1:desampled_length
        temp(:,jj) = max(x(:,jj:jj+d_scale-1),[],2); % cal the max in the 2nd dim
    end

    % 2. time/frequency series
%     x = originSet(:,ii);
%     x = abs(fft(x));
% %     result(:,ii) = x';
%     temp = zeros(32, 32);   
%     for jj = 1:d_scale
%         temp(jj,:) = x((jj-1)*32+1:jj*32); % fold the series from 2048 to 32*64
%     end
    
    x = temp; 
    xmax = max(max(x));
    xmin = min(min(x));
    x = uint8((x-xmin)/(xmax-xmin)*255);
%     for jj = 1:d_scale
%         result((jj-1)*desampled_length+1:jj*desampled_length, ii) = x(jj,:);
%     end
    
    figname = [dataname ',fft' ',rspeed-' rf ',sfre-' num2str(Fs)];
    imgname = [figname '_' num2str(ii) '.png'];
    imwrite(x,[savepath imgname],'png')
end
% figure
% imshow(uint8(result))
%}

%% save originalset
%
dataname = [dataname ',raw' ',rspeed_' rf ',sfre_' num2str(Fs)];
save([savepath dataname '.mat'], 'originSet')
%
end
