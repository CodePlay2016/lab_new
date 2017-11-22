function [y, target] = divideData(original, lengthOfEachSample, divideGap, piecesDivided, targetType)
% �˷������ڽ���ʱ�����зָ�Ϊ���ɸ���ʱ������
% input:
% |-original: ԭʼʱ������,Ϊn*N������
% |-lengthOfEachSample: ��Ҫ�ָ���ÿһ��ʱ�����еĳ���,����
% |-divideGap: �Զ���Ϊ������зָ�,����
% |-piecesDivided: �ָ�Ϊ���ٷ�,����
% |-targetType: ÿһ��ԭʼʱ�����ж�Ӧ������,����Ȼ�����������������,1*nʸ��
%               ��:����8������,ǰ4��Ϊһ��,������������һ��,��targetTypeӦΪ:
%               [0,0,0,0,1,1,2,2]
% output:

[originalNumber, length] = size(original);
numOfPieces = originalNumber*piecesDivided;

if (piecesDivided-1)*divideGap+lengthOfEachSample >= length
    piecesDivided = floor((length-lengthOfEachSample)/divideGap);
end

y = zeros(lengthOfEachSample,numOfPieces);
% digits = 4; % ����bitget�������õ���λ��
digits = max(targetType) + 1;
target = zeros(digits,numOfPieces);

for kk=1:originalNumber*piecesDivided
    % �ָ�����
    ii = floor((kk-1)/piecesDivided)+1; % ii��ʾ��ǰѭ�����ڼ���original
    istart = (kk-(ii-1)*piecesDivided-1)*divideGap+1;
    iend = istart+lengthOfEachSample-1;
    y(:,kk)=original(ii,istart:iend)';
%     t = floor(ii*(originalNumber/length(targetType))+1;
    % ���ָ�����������ǩ:�����Զ����Ƶ���ʽ
%     target(:,kk)=bitget(targetType(1,ii),digits:-1:1);
    target(targetType(ii)+1,kk) = 1;
end

