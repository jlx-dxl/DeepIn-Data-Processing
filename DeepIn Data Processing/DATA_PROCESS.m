for n=1:11
    if n<8
        sheet_list = ['A','B'];
    else
        sheet_list = ['A','B','C'];
    end
    for j=1:length(sheet_list)
        path=['320Hz_',num2str(n)];
        snr = xlsread([path,'\',path,'_snr_data.xls'],sheet_list(j),'A3:F3203');
        ssim = xlsread([path,'\',path,'_snr_data.xls'],sheet_list(j),'G3:L3203');
        psnr = xlsread([path,'\',path,'_snr_data.xls'],sheet_list(j),'M3:R3203');
        Qlist = [1,5,10,40,150,600];
        
        for i=1:length(Qlist)
        snr_mean(:,i) = mean(snr(:,i));
        ssim_mean(:,i) = mean(ssim(:,i));
        psnr_mean(:,i) = mean(psnr(:,i));
        end
        
        X=[1,1.5,2.0,2.5,3.0,3.5];
        
        figure('Visible','off');
        boxplot(snr,Qlist,'Widths',0.3,'Positions',X,'Symbol','r.');
        xlabel('Q');
        ylabel('SNR');
        title([path,'\SNR_',sheet_list(j)]);
        hold on;
        plot(X,snr_mean,'m--*');
        str1=num2str(snr_mean');text(X,snr_mean,str1)
        saveas(gcf,[path,'\',path,'_',sheet_list(j),'_SNR','.png']);
        
        figure('Visible','off');
        boxplot(ssim,Qlist,'Widths',0.3,'Positions',X,'Symbol','r.');
        xlabel('Q');
        ylabel('SSIM');
        title([path,'\SSIM_',sheet_list(j)]);
        hold on;
        plot(X,ssim_mean,'m--*');
        str1=num2str(ssim_mean');text(X,ssim_mean,str1)
        saveas(gcf,[path,'\',path,'_',sheet_list(j),'_SSIM','.png']);

        figure('Visible','off');
        boxplot(psnr,Qlist,'Widths',0.3,'Positions',X,'Symbol','r.');
        xlabel('Q');
        ylabel('PSNR');
        title([path,'\PSNR_',sheet_list(j)]);
        hold on;
        plot(X,psnr_mean,'m--*');
        str1=num2str(psnr_mean');text(X,psnr_mean,str1)
        saveas(gcf,[path,'\',path,'_',sheet_list(j),'_PSNR','.png']);

    end
end