tic
snr = xlsread('snr_Q.xls','snr','A2:GR9601');
ssim = xlsread("snr_Q.xls",'ssim','A2:GR9601');
psnr = xlsread("snr_Q.xls",'psnr','A2:GR9601');
Q_list = [1:1:100,109:9:1000];

for i=1:200
snr_mean(:,i) = mean(snr(:,i));
ssim_mean(:,i) = mean(ssim(:,i));
psnr_mean(:,i) = mean(psnr(:,i));
snr_max(:,i) = max(snr(:,i));
ssim_max(:,i) = max(ssim(:,i));
psnr_max(:,i) = max(psnr(:,i));
snr_min(:,i) = min(snr(:,i));
ssim_min(:,i) = min(ssim(:,i));
psnr_min(:,i) = min(psnr(:,i));
end

figure('Visible','off')
xlabel('Q');
ylabel('SNR(dB)');
title('SNR-Q Curve')
plot(Q_list,snr_mean,'r.-')
hold on 
plot(Q_list,snr_max,'g.-')
hold on
plot(Q_list,snr_min,'b.-')
legend('SNR-mean','SNR-max','SNR-min','Location','southeast')
saveas(gcf,['SNR-Q Curve.png']);

figure('Visible','off')
xlabel('Q');
ylabel('SSIM');
title('SSIM-Q Curve')
plot(Q_list,ssim_mean,'r.-')
hold on 
plot(Q_list,ssim_max,'g.-')
hold on
plot(Q_list,ssim_min,'b.-')
legend('SSIM-mean','SSIM-max','SSIM-min','Location','southeast')
saveas(gcf,['SSIM-Q Curve.png']);

figure('Visible','off')
xlabel('Q');
ylabel('PSNR(dB)');
title('PSNR-Q Curve')
plot(Q_list,psnr_mean,'r.-')
hold on 
plot(Q_list,psnr_max,'g.-')
hold on
plot(Q_list,psnr_min,'b.-')
legend('PSNR-mean','PSNR-max','PSNR-min','Location','southeast')
saveas(gcf,['PSNR-Q Curve.png']);
toc

