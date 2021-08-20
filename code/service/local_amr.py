"""
AMR parsing service using a locally deployed model
"""

from transition_amr_parser.parse import AMRParser

in_checkpoint = "DATA/AMR2.0/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf/ep60-seed44/checkpoint_wiki.smatch_best1.pt"


class AMRClient(object):
    def __init__(self):
        self.parser = AMRParser.from_checkpoint(in_checkpoint)

    def get_amr(self, text):
        res = self.parser.parse_sentences([text.split()])
        return res[0][0]


if __name__ == "__main__":
    """
    parser = AMRParser.from_checkpoint(in_checkpoint)
    annotations = parser.parse_sentences([['The', 'boy', 'travels'], ['He', 'visits', 'places']])
    # Penman notation
    for amr in annotations[0]:
        print("".join(amr))
    """
    amr = AMRClient().get_amr("The boy travels")
    print(amr)

