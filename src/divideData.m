function [y, target] = divideData(original, lengthOfEachSample, divideGap, piecesDivided, targetType)
% 此方法用于将长时间序列分割为若干个短时间序列
% input:
% |-original: 原始时间序列,为n*N的向量
% |-lengthOfEachSample: 想要分割后的每一个时间序列的长度,标量
% |-divideGap: 以多少为间隔进行分割,标量
% |-piecesDivided: 分割为多少份,标量
% |-targetType: 每一个原始时间序列对应的类型,以自然数对类与类进行区分,1*n矢量
%               如:共有8组数据,前4组为一种,后四组中两两一组,则targetType应为:
%               [0,0,0,0,1,1,2,2]
% output:

[originalNumber, length] = size(original);
numOfPieces = originalNumber*piecesDivided;

if (piecesDivided-1)*divideGap+lengthOfEachSample >= length
    piecesDivided = floor((length-lengthOfEachSample)/divideGap);
end

y = zeros(lengthOfEachSample,numOfPieces);
% digits = 4; % 后面bitget函数所用到的位数
digits = max(targetType) + 1;
target = zeros(digits,numOfPieces);

for kk=1:originalNumber*piecesDivided
    % 分割数据
    ii = floor((kk-1)/piecesDivided)+1; % ii表示当前循环到第几个original
    istart = (kk-(ii-1)*piecesDivided-1)*divideGap+1;
    iend = istart+lengthOfEachSample-1;
    y(:,kk)=original(ii,istart:iend)';
%     t = floor(ii*(originalNumber/length(targetType))+1;
    % 给分割后的序列贴标签:必须以二进制的形式
%     target(:,kk)=bitget(targetType(1,ii),digits:-1:1);
    target(targetType(ii)+1,kk) = 1;
end

