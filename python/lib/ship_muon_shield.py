import json
import time

import numpy as np


def CreateArb8(arbName, medium, dZ, corners, color, magField, tShield, x_translation, y_translation, z_translation, stepGeo):
    assert stepGeo == False

    tShield['components'].append({
        'corners' : corners,
        'field' : magField,
        'name': arbName,
        'dz' : dZ,
        "z_center" : z_translation,
    })


# fields should be 4x3 np array
def create_magnet(magnetName, medium, tShield,
                  fields, fieldDirection, dX,
                  dY, dX2, dY2, dZ, middleGap,
                  middleGap2, HmainSideMag, HmainSideMag2, gap,
                  gap2, Z, NotMagnet, stepGeo, SC_key=False):
    fDesign = 8

    if NotMagnet:
        coil_gap = gap
        coil_gap2 = gap2
    elif fDesign > 7:
        # Assuming 0.5A / mm ^ 2 and 10000At needed, about 200cm ^ 2 gaps are necessary
        # Current design safely above this.Will consult with MISiS to get a better minimum.
        gap = np.ceil(max(100. / dY, gap))
        gap2 = np.ceil(max(100. / dY2, gap2))
        coil_gap = gap
        coil_gap2 = gap2
    else:
        raise NotImplementedError("Design value is wrong.")

    anti_overlap = 0.1 # gap between fields in the corners for mitred joints (Geant goes crazy when
    # they touch each other)

    if not SC_key:
        cornersMainL = np.array([
            middleGap, -(dY + dX - anti_overlap), middleGap, dY + dX - anti_overlap,
            dX + middleGap, dY - anti_overlap, dX + middleGap,
            -(dY - anti_overlap),
            middleGap2, -(dY2 + dX2 - anti_overlap), middleGap2, dY2 + dX2 - anti_overlap,
            dX2 + middleGap2, dY2 - anti_overlap, dX2 + middleGap2,
            -(dY2 - anti_overlap)])

        cornersTL = np.array((middleGap + dX,
                              dY,
                              middleGap,
                              dY + dX,
                              2 * dX + middleGap + coil_gap,
                              dY + dX,
                              dX + middleGap + coil_gap,
                              dY,
                              middleGap2 + dX2,
                              dY2,
                              middleGap2,
                              dY2 + dX2,
                              2 * dX2 + middleGap2 + coil_gap2,
                              dY2 + dX2,
                              dX2 + middleGap2 + coil_gap2,
                              dY2))

        if fDesign == 7:
            cornersMainSideL = np.array((dX + middleGap + gap, -HmainSideMag,
                                         dX + middleGap + gap, HmainSideMag,
                                         2 * dX + middleGap + gap, HmainSideMag,
                                         2 * dX + middleGap + gap, -HmainSideMag,
                                         dX2 + middleGap2 + gap2, -HmainSideMag2,
                                         dX2 + middleGap2 + gap2, HmainSideMag2,
                                         2 * dX2 + middleGap2 + gap2, HmainSideMag2,
                                         2 * dX2 + middleGap2 + gap2, -HmainSideMag2))
        else:
            cornersMainSideL = np.array((dX + middleGap + gap, -(dY - anti_overlap), dX + middleGap + gap,
                                         dY - anti_overlap, 2 * dX + middleGap + gap, dY + dX - anti_overlap,
                                         2 * dX + middleGap + gap, -(dY + dX - anti_overlap), dX2 + middleGap2 + gap2,
                                         -(dY2 - anti_overlap), dX2 + middleGap2 + gap2, dY2 - anti_overlap,
                                         2 * dX2 + middleGap2 + gap2, dY2 + dX2 - anti_overlap, 2 * dX2 + middleGap2 + gap2,
                                         -(dY2 + dX2 - anti_overlap)))

    else:
        cornersMainL = np.array([middleGap,        -(dY + 3 * dX - anti_overlap),
                      middleGap,        dY + 3 * dX - anti_overlap,
                      dX + middleGap,   dY - anti_overlap,
                      dX + middleGap,   -(dY - anti_overlap),
                      middleGap2,       -(dY2 + 3 * dX2 - anti_overlap),
                      middleGap2,       dY2 + 3 * dX2 - anti_overlap,
                      dX2 + middleGap2, dY2 - anti_overlap,
                      dX2 + middleGap2, -(dY2 - anti_overlap)])
        cornersTL = np.array([middleGap + dX,
                   dY,
                   middleGap,
                   dY + 3 * dX,
                   4 * dX + middleGap + coil_gap,
                   dY + 3 * dX,
                   dX + middleGap + coil_gap,
                   dY,
                   middleGap2 + dX2,
                   dY2,
                   middleGap2,
                   dY2 + 3 * dX2,
                   4 * dX2 + middleGap2 + coil_gap2,
                   dY2 + 3 * dX2,
                   dX2 + middleGap2 + coil_gap2,
                   dY2])

        if fDesign == 7:
            cornersMainSideL = np.array([dX + middleGap + gap,        -HmainSideMag,
                                                       dX + middleGap + gap,        HmainSideMag,
                                                       4 * dX + middleGap + gap,    HmainSideMag,
                                                       4 * dX + middleGap + gap,    -HmainSideMag,
                                                       dX2 + middleGap2 + gap2,     -HmainSideMag2,
                                                       dX2 + middleGap2 + gap2,     HmainSideMag2,
                                                       4 * dX2 + middleGap2 + gap2, HmainSideMag2,
                                                       4 * dX2 + middleGap2 + gap2, -HmainSideMag2])
        else:
            cornersMainSideL = np.array([dX + middleGap + gap,        -(dY - anti_overlap),
                                                       dX + middleGap + gap,        dY - anti_overlap,
                                                       4 * dX + middleGap + gap,    dY + 3 * dX - anti_overlap,
                                                       4 * dX + middleGap + gap,    -(dY + 3 * dX - anti_overlap),
                                                       dX2 + middleGap2 + gap2,     -(dY2 - anti_overlap),
                                                       dX2 + middleGap2 + gap2,     dY2 - anti_overlap,
                                                       4 * dX2 + middleGap2 + gap2, dY2 + 3 * dX2 - anti_overlap,
                                                       4 * dX2 + middleGap2 + gap2, -(dY2 + 3 * dX2 - anti_overlap)])

    cornersMainR = np.zeros(16, np.float64)
    cornersCLBA = np.zeros(16, np.float64)
    cornersMainSideR = np.zeros(16, np.float64)
    cornersCLTA = np.zeros(16, np.float64)
    cornersCRBA = np.zeros(16, np.float64)
    cornersCRTA = np.zeros(16, np.float64)

    cornersTR = np.zeros(16, np.float64)
    cornersBL = np.zeros(16, np.float64)
    cornersBR = np.zeros(16, np.float64)

    if (fDesign == 7):
        cornersCLBA = np.array([dX + middleGap + gap,
                                -HmainSideMag,
                                2 * dX + middleGap + gap,
                                -HmainSideMag,
                                2 * dX + middleGap + coil_gap,
                                -(dY + dX - anti_overlap),
                                dX + middleGap + coil_gap,
                                -(dY - anti_overlap),
                                dX2 + middleGap2 + gap2,
                                -HmainSideMag2,
                                2 * dX2 + middleGap2 + gap2,
                                -HmainSideMag2,
                                2 * dX2 + middleGap2 + coil_gap2,
                                -(dY2 + dX2 - anti_overlap),
                                dX2 + middleGap2 + coil_gap2,
                                -(dY2 - anti_overlap)])

    # Use symmetries to define remaining magnets
    for i in range(16):
        cornersMainR[i] = -cornersMainL[i]
        cornersMainSideR[i] = -cornersMainSideL[i]
        cornersCRTA[i] = -cornersCLBA[i]
        cornersBR[i] = -cornersTL[i]

    # Need to change order as corners need to be defined clockwise
    for i in range(8):
        j = (11 - i) % 8
        cornersCLTA[2 * j] = cornersCLBA[2 * i]
        cornersCLTA[2 * j + 1] = -cornersCLBA[2 * i + 1]
        cornersTR[2 * j] = -cornersTL[2 * i]
        cornersTR[2 * j + 1] = cornersTL[2 * i + 1]

    for i in range(16):
        cornersCRBA[i] = -cornersCLTA[i]
        cornersBL[i] = -cornersTR[i]

    str1L = "_MiddleMagL"
    str1R = "_MiddleMagR"
    str2 = "_MagRetL"
    str3 = "_MagRetR"
    str4 = "_MagCLB"
    str5 = "_MagCLT"
    str6 = "_MagCRT"
    str7 = "_MagCRB"
    str8 = "_MagTopLeft"
    str9 = "_MagTopRight"
    str10 = "_MagBotLeft"
    str11 = "_MagBotRight"

    stepGeo = False

    theMagnet = {
        'components' : []
    }

    color = [0, 1, 2, 3]
    if fieldDirection == "up":
        CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
        if fDesign==7:
            CreateArb8(magnetName + str4, medium, dZ, cornersCLBA, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
            CreateArb8(magnetName + str5, medium, dZ, cornersCLTA, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
            CreateArb8(magnetName + str6, medium, dZ, cornersCRTA, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
            CreateArb8(magnetName + str7, medium, dZ, cornersCRBA, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str8, medium, dZ, cornersTL, color[3], fields[3], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str9, medium, dZ, cornersTR, color[2], fields[2], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str10, medium, dZ, cornersBL, color[2], fields[2], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str11, medium, dZ, cornersBR, color[3], fields[3], theMagnet, 0, 0, Z, stepGeo)

    elif fieldDirection == "down":
        CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, color[1], fields[1], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
        if fDesign==7:
            CreateArb8(magnetName + str4, medium, dZ, cornersCLBA, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
            CreateArb8(magnetName + str5, medium, dZ, cornersCLTA, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
            CreateArb8(magnetName + str6, medium, dZ, cornersCRTA, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
            CreateArb8(magnetName + str7, medium, dZ, cornersCRBA, color[0], fields[0], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str8, medium, dZ, cornersTL, color[2], fields[2], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str9, medium, dZ, cornersTR, color[3], fields[3], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str10, medium, dZ, cornersBL, color[3], fields[3], theMagnet, 0, 0, Z, stepGeo)
        CreateArb8(magnetName + str11, medium, dZ, cornersBR, color[2], fields[2], theMagnet, 0, 0, Z, stepGeo)
    else:
        raise RuntimeError("File direction value not recognized.")

    theMagnet['dz'] = dZ
    theMagnet['z_center'] = Z

    tShield['magnets'].append(theMagnet)


def design_muon_shield(params,fSC_mag = True):
    n_magnets = 9
    cm = 1
    mm = 0.1 * cm
    m = 100 * cm
    tesla = 10
    fField = 1.7

    magnetName = ["MagnAbsorb1", "MagnAbsorb2", "Magn1", "Magn2", "Magn3", "Magn4", "Magn5", "Magn6", "Magn7"]

    fieldDirection = ["up", "up", "up", "up", "up", "down", "down", "down", "down"]

    zgap = 10 * cm

    LE = 7 * m
    dZ0 = 1 * m
    dZ1 = 0.4 * m
    dZ2 = 2.31 * m
    dZ3 = params[2]
    dZ4 = params[3]
    dZ5 = params[4]
    dZ6 = params[5]
    dZ7 = params[6]
    dZ8 = params[7]
    fMuonShieldLength = 2 * (dZ1 + dZ2 + dZ3 + dZ4 + dZ5 + dZ6 + dZ7 + dZ8) + LE


    dXIn = np.zeros(n_magnets)
    dXOut = np.zeros(n_magnets)
    gapIn = np.zeros(n_magnets)
    dYIn = np.zeros(n_magnets)
    dYOut = np.zeros(n_magnets)
    gapOut = np.zeros(n_magnets)
    dZf = np.zeros(n_magnets)

    Z = np.zeros(n_magnets)
    midGapIn= np.zeros(n_magnets)
    midGapOut= np.zeros(n_magnets)
    HmainSideMagIn= np.zeros(n_magnets)
    HmainSideMagOut= np.zeros(n_magnets)

    offset = 7

    dXIn[0] = 0.4 * m
    dXOut[0] = 0.40 * m
    gapIn[0] = 0.1 * mm
    dYIn[0] = 1.5 * m
    dYOut[0] = 1.5 * m
    gapOut[0] = 0.1 * mm
    dXIn[1] = 0.5 * m
    dXOut[1] = 0.5 * m
    gapIn[1] = 0.02 * m
    dYIn[1] = 1.3 * m
    dYOut[1] = 1.3 * m
    gapOut[1] = 0.02 * m


    offset = 7

    for i in range(2, n_magnets-1):
        dXIn[i] = params[offset + i * 6 + 1]
        dXOut[i] = params[offset + i * 6 + 2]
        dYIn[i] = params[offset + i * 6 + 3]
        dYOut[i] = params[offset + i * 6 + 4]
        gapIn[i] = params[offset + i * 6 + 5]
        gapOut[i] = params[offset + i * 6 + 6]

    XXX = -25 * m - fMuonShieldLength / 2. # TODO: This needs to be checked
    zEndOfAbsorb = XXX - fMuonShieldLength / 2.

    dZf[0] = dZ1 - zgap / 2
    Z[0] = zEndOfAbsorb + dZf[0] + zgap
    dZf[1] = dZ2 - zgap / 2
    Z[1] = Z[0] + dZf[0] + dZf[1] + zgap
    dZf[2] = dZ3 - zgap / 2
    Z[2] = Z[1] + dZf[1] + dZf[2] + 2 * zgap
    dZf[3] = dZ4 - zgap / 2
    Z[3] = Z[2] + dZf[2] + dZf[3] + zgap
    dZf[4] = dZ5 - zgap / 2
    Z[4] = Z[3] + dZf[3] + dZf[4] + zgap
    dZf[5] = dZ6 - zgap / 2
    Z[5] = Z[4] + dZf[4] + dZf[5] + zgap
    dZf[6] = dZ7 - zgap / 2
    Z[6] = Z[5] + dZf[5] + dZf[6] + zgap
    dZf[7] = dZ8 - zgap / 2
    Z[7] = Z[6] + dZf[6] + dZf[7] + zgap

    dXIn[8] = dXOut[7]
    dYIn[8] = dYOut[7]
    dXOut[8] = dXIn[8]
    dYOut[8] = dYIn[8]
    gapIn[8] = gapOut[7]
    gapOut[8] = gapIn[8]
    dZf[8] = 0.1 * m
    Z[8] = Z[7] + dZf[7] + dZf[8]

    for i in range(n_magnets):
        midGapIn[i] = 0.
        midGapOut[i] = 0.
        HmainSideMagIn[i] = dYIn[i] / 2
        HmainSideMagOut[i] = dYOut[i] / 2

    mField = 1.6 * tesla
    fieldsAbsorber = [
        [0., mField, 0.],
        [0., -mField, 0.],
        [-mField, 0., 0.],
        [mField, 0., 0.]
    ]

    nM = 1

    tShield = {
        'magnets':[]
    }
    create_magnet(magnetName[nM], "G4_Fe", tShield, fieldsAbsorber, fieldDirection[nM], dXIn[nM], dYIn[nM], dXOut[nM],
                 dYOut[nM], dZf[nM], midGapIn[nM], midGapOut[nM], HmainSideMagIn[nM], HmainSideMagOut[nM],
                 gapIn[nM], gapOut[nM], Z[nM], True, False)

    fieldScale = np.ones(9)
    for nM in range(2, n_magnets):
        if (dZf[nM] < 1e-5 or nM == 4) and fSC_mag:
            continue
        ironField_s_SC = fField * fieldScale[nM] * tesla
        SC_key = False
        if nM == 3 and fSC_mag:
            SC_FIELD = 5.1
            ironField_s_SC = SC_FIELD * fieldScale[nM] * tesla
            SC_key = True
        ironField_s = fField * fieldScale[nM] * tesla
        magFieldIron_s = [0., ironField_s_SC, 0.]
        RetField_s = [0., -ironField_s, 0.]
        ConRField_s = [-ironField_s, 0., 0.]
        ConLField_s = [ironField_s, 0., 0.]
        fields_s = np.array([magFieldIron_s, RetField_s, ConRField_s, ConLField_s])

        create_magnet(magnetName[nM], "G4_Fe", tShield, fields_s, fieldDirection[nM], dXIn[nM], dYIn[nM], dXOut[nM],
                  dYOut[nM], dZf[nM], midGapIn[nM], midGapOut[nM], HmainSideMagIn[nM], HmainSideMagOut[nM],
                  gapIn[nM], gapOut[nM], Z[nM], nM == 8, False, SC_key)

    return tShield


def get_design_from_params(params, z_bias=50., force_remove_magnetic_field=False, fSC_mag:bool = True):
    mag_unit =  10.000000
    # nMagnets 9

    shield = design_muon_shield(params, fSC_mag)
    # print(shield)

    magnets_2 = []

    max_z = None
    for mag in shield['magnets']:
        mag['dz'] = mag['dz'] / 100.
        mag['z_center'] = mag['z_center'] / 100. + z_bias
        components_2 = mag['components']
        # print(components_2)

        if force_remove_magnetic_field:
            multiplier = 0
        else:
            multiplier = 1/mag_unit

        components_2 = [{'corners': (np.array(x['corners']) / 100.).tolist(),
                         'field': (x['field'][0] * multiplier, x['field'][1] *multiplier, x['field'][2] *multiplier)} for x
                        in components_2]
        mag['components'] = components_2
        mag['material'] = 'G4_Fe'
        mag['fieldX'] = 0.
        mag['fieldY'] = 0.
        mag['fieldZ'] = 0.
        magnets_2.append(mag)

        new_mz = mag['dz'] / 2 + mag['z_center'] + 0.05#limit in 31.5
        if max_z is None or new_mz > max_z:
            max_z = new_mz

    shield['magnets'] = magnets_2

    # print(shield)
    shield.update({
        "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 11, "worldSizeY": 11,
        "worldSizeZ": 100,
        "type" : 1,
        "limits" : {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        },

        "sensitive_film": {
            "z_center" : new_mz,
            "dz" : 0.01,
            "dx": 3,
            "dy": 3,
        }
    })


    return shield