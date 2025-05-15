function Compute_and_print_error_metrics(AM_Prices,BE_Prices,AM_XVAs,BE_XVAs,GPR_XVAs,Times)

GAP_AM=AM_XVAs-GPR_XVAs;
GAP_BE=BE_XVAs-GPR_XVAs;
RGAP_AM=(AM_XVAs-GPR_XVAs)./AM_XVAs;
RGAP_BE=(BE_XVAs-GPR_XVAs)./BE_XVAs;
RpGAP_AM=(AM_XVAs-GPR_XVAs)./AM_Prices;
RpGAP_BE=(BE_XVAs-GPR_XVAs)./BE_Prices;

T_no=Times;

MAE_AM=mean(abs(RGAP_AM))*100;
MAE_BE=mean(abs(RGAP_BE))*100;
RMSE_AM=sqrt(mean(GAP_AM.^2))*100;
RMSE_BE=sqrt(mean(GAP_BE.^2))*100;
RMSRE_AM=sqrt(mean(RGAP_AM.^2))*100;
RMSRE_BE=sqrt(mean(RGAP_BE.^2))*100;
RMSRpE_AM=sqrt(mean(RpGAP_AM.^2))*100;
RMSRpE_BE=sqrt(mean(RpGAP_BE.^2))*100;

MT=mean(T_no);
Lci_T=1.96*std(T_no)/sqrt(length(T_no));

fprintf("|| MAE %5.3f %5.3f ",MAE_AM,MAE_BE );
fprintf("| RMSE %5.3f %5.3f ",RMSE_AM,RMSE_BE );
fprintf("| RMSRE %5.3f %5.3f ",RMSRE_AM,RMSRE_BE );
fprintf("| RMSRpE %5.5f %5.5f ",RMSRpE_AM,RMSRpE_BE );
fprintf("AT %3.0f+-%2.0fs\n",MT,Lci_T);

end