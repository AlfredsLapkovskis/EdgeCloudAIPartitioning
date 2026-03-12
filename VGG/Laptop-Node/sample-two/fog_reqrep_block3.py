from fog_receiver_reqrep_common import run_fog_receiver_reqrep

PORT = 5558

if __name__ == "__main__":
    run_fog_receiver_reqrep(
        bind_port=PORT,
        expected_split_name="split_after_block3",
        split_index=17,
        output_csv="laptop_block3_emissions.csv",
    )
