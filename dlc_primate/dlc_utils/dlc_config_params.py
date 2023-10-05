config_edits_face = {
    'dotsize': 3,  # size of the dots!
    'colormap': 'rainbow',  # any matplotlib colormap
    'alphavalue': 0.2, # transparency of labels
    'pcutoff': 0.5,  # the higher the more conservative the plotting!
    'skeleton': 
         # Right Eye
        [['RightEye_Top', 'RightEye_Inner'], 
         ['RightEye_Inner', 'RightEye_Bottom'],
         ['RightEye_Outer', 'RightEye_Bottom'],
         ['RightEye_Top', 'RightEye_Outer'], 
         # Left Eye
         ['LeftEye_Top', 'LeftEye_Inner'],
         ['LeftEye_Inner', 'LeftEye_Bottom'],
         ['LeftEye_Outer', 'LeftEye_Bottom'],
         ['LeftEye_Top', 'LeftEye_Outer'],
        #  # Top of Head Counter-Clockwise to Lip
        #  ['HeadTop_Mid', 'OutlineRight_Mouth'],
        #  ['OutlineRight_Mouth', 'RightNostrils_Bottom'],
        #  ['RightNostrils_Bottom', 'UpperLip_Centre'],
        #  # Lip Counter-Clockwise to Top of Head
        #  ['UpperLip_Centre', 'OutlineLeft_Mouth'],
        #  ['OutlineLeft_Mouth', 'LeftNostrils_Bottom'],
        #  ['LeftNostrils_Bottom', 'HeadTop_Mid'],
        ],
    'skeleton_color': 'white'
}