"""A wrapper using the GRPC AMR parsing service"""
from code.grpc_clients import AMRClientTransformer

amr_host = "mnlp-demo.sl.cloud9.ibm.com"
amr_port = 59990
amr_client = AMRClientTransformer(f"{amr_host}:{amr_port}")


if __name__ == "__main__":
    text = "The quick brown fox jumped over the lazy moon."
    amr = amr_client.get_amr(text)
    print("\ntext:", text)
    print("\namr:", amr)

