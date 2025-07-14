#!/bin/bash

make_job_file () {
    mkdir -p jobs  # Create jobs/ directory if it doesn't exist
    jobname=$1
    exp=$2
    cat << EOF > "jobs/job_test_${1}.sh"
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=12:00:00

python -u /Users/mathieuisoard/Documents/kaggle-competitions/CMI-sensor-competition/github/codes/DL_model.py ${@:3}  > logs/job_exp${exp}_${jobname}.log 2>&1 
EOF

    chmod +x "jobs/job_test_${1}.sh"
}

run_job () {
    curdir=$(pwd)
    host=$(uname -a | cut -d ' ' -f 2)

    if [[ "${host}" == "babur" ]]; then
        sbatch "jobs/job_test_${1}.sh"
    else
        bash "jobs/job_test_${1}.sh" > /dev/null 2>&1 &  #> /dev/null 2>&1 &
    fi
}


# -------------- Usage Example --------------

mkdir -p logs  # Optional: create a logs directory for outputs

number_exp=22

feature_sets=(
    # "acc_x acc_y acc_z rot_x rot_y rot_z rot_w"
    # "acc_x acc_y acc_z rot_x rot_y rot_z rot_w rotvec_x rotvec_y rotvec_z"
    # "acc_x acc_y acc_z rot_x rot_y rot_z rot_w rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z"
    # "acc_x acc_y acc_z rot_x rot_y rot_z rot_w rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z"
    # "acc_x acc_y acc_z rot_x rot_y rot_z rot_w linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z phase_adj"
    # "acc_x acc_y acc_z rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z phase_adj"
    # "acc_x acc_y acc_z acc_norm rot_x rot_y rot_z rot_w rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z linear_acc_norm ang_vel_x ang_vel_y ang_vel_z phase_adj"
    # "acc_x acc_y acc_z rot_x rot_y rot_z rot_w rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist phase_adj"
    "acc_x acc_y acc_z rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist phase_adj" #---> BEST ATM
    #"rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist phase_adj"
    #"acc_x acc_y acc_z rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist rot_angle phase_adj"
    #"acc_x acc_y acc_z rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist rot_angle_vel phase_adj"
    #"acc_x acc_y acc_z acc_x_FFT acc_y_FFT acc_z_FFT rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist phase_adj"
    #"acc_x acc_y acc_z rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z linear_acc_x_FFT linear_acc_y_FFT linear_acc_z_FFT ang_vel_x ang_vel_y ang_vel_z ang_dist phase_adj"
    #"acc_x acc_y acc_z acc_x_FFT acc_y_FFT acc_z_FFT rotvec_x rotvec_y rotvec_z rotvec_x_FFT rotvec_y_FFT rotvec_z_FFT linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist phase_adj"
    #"acc_x acc_y acc_z acc_x_world acc_y_world acc_z_world rotvec_x rotvec_y rotvec_z linear_acc_x linear_acc_y linear_acc_z ang_vel_x ang_vel_y ang_vel_z ang_dist phase_adj"
)


for i in "${!feature_sets[@]}"; do
    make_job_file $i $number_exp ${feature_sets[$i]}
    run_job $i
done

# # Submit jobs
# run_job "job1"
# run_job "job2"
# run_job "job3"
