#!/bin/bash

# Check if target path is provided
if [ $# -lt 1 ]; then
    echo "Error: Please provide a target directory to download files."
    echo "Usage: $0 /path/to/output_directory"
    exit 1
fi

OUTPUT_PATH="$1"
mkdir -p "$OUTPUT_PATH"

# List of impulse responses
IRS=(
    IR_AKGD12.wav
    IR_AKG_FaultyD12.wav

    Altec_639.wav
    Altec_670A.wav
    Altec_670B.wav

    American_R331.wav

    Amperite_RA.wav

    IR_Astatic77.wav

    B%26O_BM2.wav
    Beomic_1000.wav
    B%26O_BM6.wav

    BBCmarconi_B.wav

    IR_BeyerM500Stock.wav
    Beyer_M360.wav
    Beyer_M260.wav

    Coles_4038.wav

    Doremi_.wav

    EV_RE20_Flat.wav
    EV_RE20_HPF.wav

    EMI_ribbon.wav

    FilmIndustries_M8.wav

    IR_FosterDynamicDF1.wav

    IR_Meazzi.wav

    IR_GaumontKalee.wav

    GEC_bigdynamic.wav
    GEC_2373.wav

    Grampian_GR2.wav

    IR_Lomo52A5M.wav

    IR_MelodiumRM6.wav
    Melodium_Model12.wav
    Melodium_42B_1.wav

    IR_OktavaMD57.wav
    IR_OktavaML16.wav
    IR_OktavaMK18Silver.wav
    IR_OktavaMK18_Overload.wav
    Oktava_ML19.wav

    RCA_KU3a_1.wav
    IR_RCAKU3a.wav
    RCA_PB90.wav
    RCA_74B.wav
    RCA_77DX_1.wav
    RCA_77DX_2.wav
    RCA_44BX_1.wav
    RCA_44BX_2.wav
    RCA_varacoustic_fig8.wav

    IR_ResloDynamic.wav
    IR_ResloURA.wav
    IR_ResloCR600.wav
    Reslo_RB250.wav
    Reslo_RB_RedLabel.wav
    Reslo_SR1.wav
    Reslo_RV.wav
    Reslo_VMC2.wav

    IR_Shure510C.wav
    Shure315_flat.wav
    Shure315_HPF.wav

    Sony_C37Fet.wav

    IR_STC4035.wav
    IR_STC4033_Cardioid.wav
    IR_STC4033_Ribbon.wav
    IR_STC4033_Pressure.wav
    Coles_4038.wav

    Telefunken_M201.wav

    Toshiba_TypeG.wav
    Toshiba_BK5.wav
    Toshiba_TypeK_flat.wav
    oshiba_TypeK_HPF.wav

    GEC_bigdynamic.wav

    Altec_639.wav
)

AZURE_URL="http://xaudia.com/MicIRP/IR"

for IR in "${IRS[@]}"
do
    URL="$AZURE_URL/$IR"
    echo "Downloading $IR ..."
    curl -s -o "$OUTPUT_PATH/$IR" "$URL"
done

echo "Download complete. Files saved to: $OUTPUT_PATH"
