import pandas as pd

PHENIXPC2CH_list = [
    ('PC1', 'Ch1', '1'),
    ('PC1', 'Ch2', '2'),
    ('PC2', 'Ch1', '3'),
    ('PC2', 'Ch2', '4'),
    ('PC3', 'Ch1', '5'),
    ('PC3', 'Ch2', '6')
    ]

PHENIXPC2CH_DF = pd.DataFrame(PHENIXPC2CH_list, columns = ['PC_number', 'PC_channel', 'camera'])
