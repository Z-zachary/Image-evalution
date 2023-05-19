Metric_list = ['EN', 'MI', 'SF', 'AG', 'SD', 'CC', 'SCD', 'VIF', 'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM']
for metric in Metric_list:
    print('print(\'{}:\',  round({}, 4))'.format(metric, metric))