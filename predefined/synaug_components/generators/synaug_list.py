import os


def returnsynaug() -> list:
    synaug = [
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3608_EAC\20140121\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3621_EAC\20130812\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3570_EAC\20130927\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3581_EAC\20131025\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3101_EAC\20110330\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3043_EAC\20110214\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3065_EAC\20110214\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3205_EAC\20110920\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3235_EAC\20111028\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3281_EAC\20120207\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3306_EAC\20120220\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3387_EAC\20120705\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3354_EAC\20120507\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3696_EAC\20140114\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth2900_EAC\20101111\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth2848_EAC\20100824\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3439_EAC\20120917\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3677_EAC\20131119\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3374_EAC\20120612\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3324_EAC\20120410\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3787_EAC\20140610\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3210_EAC\20111004\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth2850_EAC\20100824\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3568_EAC\20130624\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3327_EAC\20120416\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3598_EAC\20130712\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3589_EAC\20130607\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3803_EAC\20140702\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3327_EAC\20120416\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3598_EAC\20130712\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3589_EAC\20130607\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
        r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3803_EAC\20140702\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis',
    ]
    return synaug


def returnsynorigin(path_synaug:str) -> str:
    # r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3803_EAC\20140702\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis'
    origin = os.path.dirname(os.path.dirname(os.path.dirname(path_synaug)))  
    # r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3803_EAC\20140702\RightWrist_PostTRAT1f
    origin = os.path.join(origin, 'images')
    if os.path.exists(os.path.join(origin, 'mha')):
        origin = os.path.join(origin, 'mha')
    else:
        origin = os.path.join(origin, 'itk')
    filename = os.listdir(origin)
    return os.path.join(origin, filename[0])


def returnsynauglist(path_synaug:str) -> list[str]:
    # r'R:\ESMIRA\ESMIRA_Database\LUMC\ESMIRA_patient_Arth3803_EAC\20140702\RightWrist_PostTRAT1f\results\augmentation\tenosynovitis'
    filelist = os.listdir(path_synaug)
    synaug = [os.path.join(path_synaug, item) for item in filelist]
    return synaug


def createcombination():
    synaug_list = returnsynaug()
    synaug = []
    for path in synaug_list:
        origin = returnsynorigin(path)
        auglist = returnsynauglist(path)
        for aug in auglist:
            synaug.append([origin, aug])
    return synaug  # [N*[origin, aug]]
