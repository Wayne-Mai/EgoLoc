import os
import sys
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

# sys.path.append('./Camera_Intrinsics_API/')
# from extract_frames import FrameExtractor


import os
import av

class FrameExtractor():
    def extract(self, filename: str, output_dir: str) -> None:
        reader = av.open(filename, 'r')
        for i, frame in enumerate(reader.decode(video=0)):
            image = frame.to_image()
            image.save(os.path.join(output_dir,
                                    "color_{:07d}.jpg".format(i)
                                   )
                      )

my_json={"train": ["8e525a99-786e-480e-93c9-feacfdf4ec61", "358ec512-524b-42ef-9b10-1df4ddad5661", "6b7ceb0a-f012-4506-bc4e-a50d7536b9ef", "0c08ebfa-5373-457c-8bc6-6f5c267e03b3", "95acbace-bd51-4d60-bc66-814945588be0", "22ffbd28-748d-47ec-b2f3-9b58f590285e", "94827d75-45cb-46e1-9c5d-8f2c991849ee", "4fe76a25-2a6d-44bb-ab28-2dcf796572cd", "507cf5d4-152a-4e8a-8407-d485f6e35b88", "60faf711-ca36-4c72-8ed3-49755ce45a56", "83900c13-afc6-4911-8a08-371a6be0cdb4", "43f2b0fa-ca53-4616-89a7-c50dd06cc281", "b11d6270-aedb-4cf7-9785-cef161d04c63", "8f830ea8-da9f-4552-a519-84cd9c819fe4", "4189532b-10c0-4d17-b084-45fefc6f77a7", "f7f4171d-745b-491e-875c-221b918617cc", "c7653a1d-2645-4f39-9a6f-73c834a23739", "94978ed1-1d58-498c-96b5-e9b29410e01b", "d77b9d16-1cda-4a75-ac25-c61964ea2fe8", "e008ffbb-1fd0-4cf1-a9d3-690542ef21b5", "a22e3deb-69a6-443f-83bb-406445e9062a", "ae65eb84-db95-4912-97fc-93dcba293932", "61576b93-d1a6-4fd1-806e-67811428f349", "990c1b63-5864-47bd-b45c-44b7a616c2d7", "adeb8af8-5ad5-41d3-bc17-d8d06a8d05d7", "83ca2bb7-5674-4bde-af9b-2056bdb3ac8d", "ebd43cf4-ee34-411d-9d28-ce8afe28f882", "7dbcc836-cf84-4093-9600-bc71a8e69fa2", "91647c85-6673-4be8-a12d-9a72046a2996", "17378d44-a657-4384-891c-346b256bbe34", "1ffcb567-c864-4611-84bb-d4d76f2405f5", "eb88d9fb-b35c-4b09-bff8-1f97f3e315c5", "94d3e48c-cff8-4a50-ae68-d4b49e226a49", "b6a6c500-5e85-4699-8169-5e3e2528c838", "2d129b76-232b-4df4-a885-c279be452d82", "dfad5aaf-eca9-4424-833d-2a35efd3d3d1", "cdd6616c-a0e6-4c46-81d1-1f9d4c42dda8", "559df73c-5382-4fd1-9d39-9e2c5b0cd23a", "5d9ab95a-a9a7-443f-a6cc-4cce8d45fb3e", "69da2eca-e152-4366-9a1c-ca53c2b87f23", "6acb2e7b-1ac6-42a6-8187-66755223d0c8", "b6eaea86-7140-4151-9f8d-54185a3ca4d6", "503d5009-1300-4963-98fc-e026043e162a", "c72dce47-2ba7-4840-ae05-e23139bbb841", "1b79cd27-02e5-4a36-9d51-9c9c93d30d1d", "354f0f61-f347-4434-a3d2-68550ba2d4c9", "b8132cc3-21aa-490c-919f-4a493b865c24", "dc3736a4-6620-4d5f-a9f8-701be0f4ff17", "a0799dcb-b14e-48d4-819a-4ee882bccb87", "ae1c7722-2162-4557-a5a0-15e4b4a2bf5d", "4c6d1568-70f5-4e3b-9906-d6a5f34bd698", "a40d9c7a-6b43-4837-be7b-15190838475d", "7ebe4e31-36b8-4e5b-8efb-db35397151b9", "9de6abeb-4566-4ea5-b4b7-1c825e1edc52", "f08f7975-bb9e-4ef2-989f-29de6949ddeb", "79a7887f-d765-4aea-adf3-526f201ac226", "d6ada08c-d359-4f18-85e8-fa7635c3b347", "1673a05f-d422-415e-8e76-d6dc0f80246d", "df127353-03e4-44dd-a973-91aed209120a", "98d97751-3415-42d4-a83c-6a8801554b97", "d0b0c9ad-09ea-4680-b493-5e7435412754", "286f9de0-8e55-4403-a119-65e88748480b", "487ad389-d35c-4b05-a8e6-9f78df10d7d9", "615200e4-9ed5-4f72-819f-0e6e4216f67f", "a501037b-d904-460b-8101-3c9cd936681f", "b7ff9c3c-6601-4597-b9bf-2da4dfce344a", "30c0d003-96f6-4adb-93a7-b9090c3b2300", "4d2e760d-f390-48b4-9e11-76cf04c15583", "e55b572f-a526-4fcf-88c0-3dfa5e7090b9", "377d4958-9e16-43b8-9ec4-041bee39ab5c", "172e5a39-084d-4715-915b-51327e15d05c", "eccfd4b3-78be-4a72-8918-696b5e83a13a", "07d51556-b587-4a2f-9e3a-e927f17099e9", "efb57fdb-d4df-490b-9bef-bc8bf8128a04", "654f6ab4-3602-4e2c-bc3e-f7ace2daac36", "932fdaa7-f53d-47b2-a26d-6cbaa078f54a", "ad789bf8-4bf4-49d8-8685-d4f093c7baef", "c1d773d7-fcdb-4d34-878a-2791c23af9d5", "900a437a-e67a-40de-95c2-84f2ff3b4d2d", "75673c36-7622-40c4-bfd8-d65b580e07e6", "f70fe5cc-5ac1-4ff2-94ab-8dd44e5db5f0", "53cd2834-1592-44b1-a224-c3c11ce824a5", "8d2fb416-3cc0-48b2-b761-014e848c31ea", "0c4ecbde-0957-4cfe-9ae2-532d0f23eda3", "c0007b04-7682-4848-8f2a-26144d16eecc", "1b160124-9552-44be-99c8-43f16d112a41", "afef7677-c5b0-48ef-ac7c-65ab53b2237d", "0c24b196-6dc8-44f5-b365-acc16ebd494d", "23bbce66-5157-48c6-8592-3e565504f86f", "acf1942e-74a1-4a69-b480-bd8c8573cbd1", "463f5337-19ae-46d8-afa4-af0da97ffddb", "e85ba9f4-352d-45f1-bf01-dcfd69772c03", "993d1c8e-f8fe-4398-8c5c-d5f906d8607e", "446cf8e8-61c5-4dc4-b48b-f9c616920bec", "f34125d8-6d39-4aad-bf00-55c62fc729f1", "1c89109e-80d1-4093-8aa2-81559da4ce7b", "69c40eb5-5260-48ae-8f4a-df54ca8e1380", "ff6dc7c8-17ad-45b6-8f84-045409dfffe2", "952a6bf1-bf81-4f38-83d7-142f97aaf172", "099fa5c9-7ff8-4814-8283-8841bc652aff", "019061c0-6d2d-4a35-a972-b6224ee38f49", "6a5d9dc9-63b1-43a2-803f-0f68ad7dc2f5", "c4a2cb40-d05b-4818-9d8e-da2633043d2c", "138a1b88-d7ea-435c-9dcf-319e0fd6f073", "500af7c6-bbd2-4f2f-86d0-0c10cb6f67eb", "b208fe4d-d4b1-4c5f-ae89-4b6a93d759e2", "f2044a58-0a77-43d7-85b3-bfedf2f66c09", "9a27d4b0-a950-43df-a8bb-3177260a48da", "94e90faf-8de6-47bf-86e1-6a05c090d9db", "94d37084-c200-439b-84ab-28dc95b37388", "26c315c1-8a58-49e3-934a-c7e11b74ffe1", "5dc56f6b-704f-4b09-a23f-c1e172badf51", "e87d3ed4-7ffd-4e46-b724-ef3929e66261", "66e37e80-c845-4591-bd38-d5d971b206dc", "0796f997-d027-4aff-b0c8-a11ddd08b4be", "524a9cd6-1637-4da2-8a51-8942c8831207", "74bc2257-8ff9-43be-9a37-20657288675d", "cd7e0800-a530-4c64-8e04-24ebe887e676", "02050a6c-29aa-47a1-bd4f-909a3c507320", "3f9ce77b-d99b-4312-aa18-5b2fe301f1b1", "d3f790a9-1145-4b66-802f-80c6cfe7e61c", "0a678fea-9136-48a4-9b8d-a656fdc4f671", "2082350d-b7d7-4833-91f4-5dd9c492a21f", "4273b0e9-e871-42b2-ad83-ece59d24a5f1", "22d9f13a-4de8-4ea5-8fb4-2ddd26c26023", "f8fade62-e630-443c-94a5-4c95512b0083", "fde80021-195f-43f8-b616-a751b45435cd", "34a6b28b-6396-4462-be94-d5c276c43965", "1b3392eb-0751-4734-bdfa-330ec988816c", "6d59384c-a5e6-4a09-bb55-bcfb2d4a660c", "28eade70-81ac-4004-ac18-bfb0a92bbfaf", "1202de14-bdc1-44bb-abd1-031b92903c42", "248d42c3-70ed-4111-b292-eedb99c088c0", "e7c55830-dd2c-4ed1-8659-108092e9e3ff", "5eb05107-300f-4203-bad7-fb6ccf05725d", "774af66d-537c-4aad-a956-076883b65c64", "840259ab-bb2f-43ad-860e-87eed878013f", "5650d9ef-3271-47a5-b01b-cde09f86f292", "aad12dd8-c0f2-4281-b5b6-87814bc0312a", "53cd1010-8b2a-4b52-8174-77fa45c88fdc", "9b540829-1907-4fc1-9c36-d61488f4ba19", "a493e0a6-15d6-4a58-b762-5cc1c76abe17", "ef7f97bc-ab9b-47ff-81da-a0d0ad82e3f3", "ec21be6d-f8a2-4112-bc26-a1fc620af9cc", "1267487c-249c-45a9-ac7e-8d729c4365d5", "19d37e19-fdfd-4f45-947c-d6d4ce0d0988", "9694dabe-c2b8-4e45-8a44-367287c1d5b9", "4495c013-b711-4676-a183-ccea758fda99", "3ec9a814-2977-4aba-8bda-014de78c886f", "111d203c-e9bc-4b19-bc01-6f087b0cc12c", "1953891d-6925-44b7-893c-27e1fa3c3696", "3622e30d-09de-4884-a826-0f90806f3401", "226c36e8-9934-4eb1-afc2-3c7a1db2cd87", "10fe1e0a-bcd9-4bcb-9179-e9efc881aef6", "2bcbd205-a879-4998-8f64-e0f3c676ed3d", "685daa52-41c3-43ff-a2d7-f8783370faf2", "f8c258cc-a304-4791-8880-0ae40980f476", "c2988888-ddb1-4b03-8f92-07090058f5b5", "06fe9739-5895-41f4-8bc5-3dab2af8644e", "83cfab12-27dc-4e00-8b2f-f040e7eddb13", "ac286cec-6b2d-4c66-8602-2f78ab0cf07b", "81d5bdb0-1cd2-4ecf-93b9-f0a0d9c68604", "b29961ee-ee24-4afb-8b25-fbf5117bc73d", "6063ec2e-55df-425a-a2c7-56e66caf1b01"], "val": ["6c641082-044e-46a7-ad5f-85568119e09e", "db6e91d9-3006-46dd-af29-b74f725fe284", "f0f5f45c-1576-4408-8ed0-a80747a51bd5", "bd365d25-88d0-4e4b-868a-59a01b134a42", "307c3ec6-886e-4d25-9ef7-7bea3cf7a243", "33d909a9-e570-4ba3-ab66-31cf66d42d6f", "ed808fab-fdda-4514-a00f-2fb87a7208ba", "66db02c4-053b-41ab-8b9f-f467ad1df560", "f9cbfb3a-d2e2-4f30-9c01-ee341b5e8887", "42265e2f-1d83-4ad3-a466-085c64736d76", "b7d5cc1a-d5fe-49a6-9243-dea23a61b5e4", "a044eb14-0f57-4128-9be8-4ab6fed911cf", "3d2ad5e6-3fae-440e-a8ca-0758c2f9245b", "9cfc44fe-5f79-46a6-8a4b-77ea495f3ef3", "9a24e436-4b41-4316-8484-7b9e084d9ca5", "1d2b62ed-fc94-4fb0-b547-8c50533b0b29", "fe3000d6-d857-4cd1-a927-0a91a7f9616e", "72154f42-399c-4f95-a667-e26528d1f52a", "b878975c-7136-48b5-aa4a-23950a604354", "87499bfb-3dcf-46fa-aef3-204fcc0496e0", "ba779301-79f4-4a7b-bb23-f7fbff508c5c", "f91c7008-f21c-4047-bfc1-d937787665e5", "f1350714-1aa5-421b-a268-690110ca9c06", "150bc185-821c-487f-b39c-4c5a0d52467d", "fe3d5a3a-a878-4b62-b99c-d1427eff1705", "5a7efdba-6701-4aaa-bff6-fbab6b293ed9", "1839af1c-6999-4bc4-87f7-550dd9be4b91", "dbabe8c7-ceea-4042-b5a2-575d2223dfd7", "b34643f2-eb18-492c-a446-f18c02c4198f", "cdac7d6f-1cb5-4f6a-9a6d-0213e055827c", "75a9383b-09d7-49bd-81c8-2a9880da6db3", "159ac0ad-e5ec-449d-a69e-556655f79b2a", "dfe962ab-6aa7-4888-9378-796817a30ab6", "b3d7a127-2aa6-42d2-88ce-fb023625117e", "d5935c29-1b8d-417d-9bbb-5ebd47e9256d", "6836bfae-e56b-457d-bade-1160256d4d9f", "1eb995af-0fdd-4f0f-a5ba-089d4b8cb445", "89f8409e-483a-432f-9f42-6c96ecd07b76", "e30fa220-83a0-4601-938e-6acb14211b72", "8d9b71b3-da22-41d2-bc2f-a0a1a6fd9c8a", "efe7dfaa-07e0-42d5-85eb-1e6f93409ab5", "477894c0-9bec-4271-9852-6d28bc05abfe", "ad9083b3-9b0b-4c27-a941-38f57be13edf", "151a8f90-fdec-4693-89d4-1a6e6459a612"], "test": ["c1d2ee0f-a206-471a-84ba-55015139baf8", "a70bea55-d052-4650-b0e3-9f2e9ec69700", "ed33c3c9-2e63-47ec-a3d8-0408910a1bef", "3eb5afd6-4adf-4801-b53a-d2090251d878", "c474b461-e727-429f-b253-ac7ebc10097a", "2337fc5e-4798-45a9-af6e-64476df3be46", "cac8a35c-026b-4c17-8f95-9cfbcfca1979", "e8c55e7d-8767-43fd-b74f-01ffaa9f02f8", "17940429-cddd-4146-a7be-d50c815919fe", "95ebdc21-8a19-4d5b-b620-eeb082362fe0", "058464f1-2c8d-4c71-8e76-bbc053a9f1b3", "b69ae0ef-cb54-43a1-bc66-84449063f99e", "d407ae63-1969-4593-adde-c88ee56b0eb4", "e806ab3b-b1c0-4d2e-9055-a04a22630963", "ab35e7a4-ff70-4eb3-a82d-03e00e6c601c", "1ee9157c-60da-4f7a-a573-938e609bab6d", "d51751f0-300a-41f0-9b69-5aa9cf04379a", "bbad791f-88e7-4163-8c42-c6c8364d3221", "759daf88-1e43-4f09-8486-725ca9d8971d", "20b16edf-9cdd-440b-9ebf-811d0e2647ac", "fa2ef57b-25cc-41a8-9acc-16497a7cc206", "0b8b281c-d26b-4059-9bb5-7ea96e57aa6e", "71242aba-94a8-48dc-a20d-9c0e15880031", "2a1f16d0-c365-4c02-ac28-845d12e6937c", "de80ae71-3857-4b04-87b8-1c6ad4662a72", "8ad65552-a6e2-4942-b245-12d60c3c6d23", "2e4044a3-23a8-4116-b38a-7df6fa9bab5b", "cfb10a75-e20d-444c-a23e-aaff7c9a85bd", "296c2734-64bd-44c0-994f-7558d6d872bc", "58b9bc16-f59a-494a-b4c0-c4eb6d030d74", "598eb006-db55-4fca-9150-4d143bf22849", "39afe671-501e-4937-818e-a6e4a6097bfc", "21f0a7b1-f613-4021-ae2d-a13e3ad681a1", "71de512d-417a-44d5-ad37-0bed8ce309ce", "05725fca-9b6c-43b1-b20f-8e05ab01824a", "8cdf4663-b3ce-4795-84d7-12be2f939fc7", "679d0e71-9e19-4a00-8feb-1c07b021a469", "0260c617-dc20-415f-a0fd-436dd7058016", "c0b6deed-de35-43d0-b83f-1cdccd0461ad", "e6bf00da-3164-4eba-9408-be90db0838af", "44762176-18a4-4a27-9478-0cd386c92eee", "5e3785e3-5c83-4ace-8fbb-5fafb16e708e", "20aee5ce-0d40-462a-bfed-8053b2a097c7", "2f9c862d-5607-4a07-b5ad-b9a2636f5053", "f81268f3-02f3-4be9-a79b-18598180ac97", "93618381-50bd-4f3d-899c-a37844966a5f", "634a5fae-5d81-460e-b82b-b5aaa4d77cb3", "2062924c-434a-4bb6-845c-5d3dfa4798c8", "13f84b98-d741-4631-93b5-a647f1d055f3", "6d78e2d3-a06b-4a0c-9afc-b04f2e9bf225", "0f52e4ce-0752-473c-bcb6-fe4dbac43b19", "a02ca803-1c12-4cfc-a502-b8a508e16bc7", "9850388d-567c-4463-91f2-5abe0abded76", "81c0ebb5-53e4-450b-a9e6-d04803c3014e", "a97419c5-7e52-42a3-851c-632b29324481", "fe71c340-80b3-42f2-8f41-e8d779731e14", "e682e089-41ef-480e-96e5-ae381742b039", "1bcb2c95-018b-4bd7-a54a-549a0e687d88", "ac4a0311-ab6a-44c0-ab28-baff98e78878", "dc5ac6b6-64a6-4673-be16-d4a2a1059aea", "321438d9-df74-41db-a7fa-478c411b63f3", "62a39ddf-a093-4be4-8ba9-1cc6e250d242", "e677256c-21dc-41c1-ab2d-ebd598cfe9be", "c16d8009-4518-4b08-9f77-9de382ae4dfd", "88a4a1a2-c61f-4a66-991d-a0d0538b0fe4", "4788e9ec-cb06-4002-82c5-f0e3adb8f37d", "b0bc5753-96c9-42bb-adb2-89e8c6b9e89a", "5b5a2203-d229-4120-b3f7-f1db3b8aaa91", "85c440c7-a8c4-4501-b7f0-aaa4913b0e1d"]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips_json",
        type=str,
        default='',
        help="a json file with all clip uids to be parsed for each split split, now we use hard encoded as default",
    )
    parser.add_argument(
        "--split",
        type=str,
        default='',
        help="split set to process. If not specified process all clips.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips/',
        help="Input folder with the clips.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/clip_frames/',
        help="Output folder with the clips.",
    )
    parser.add_argument(
        "--j",
        type=int,
        default=32,
        help="Number of parallel processes",
    )
    args = parser.parse_args()

    clip_dir = args.input_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    clips_filenames = os.listdir(clip_dir)
    # if args.clips_json:
    clips_json = my_json
    # if args.split:
    #     split = args.split
    #     clips_filenames = [x for x in clips_filenames if\
    #                         x.split('.')[0] in clips_json[split]
    #                         ]
    #     num_clips = len(clips_filenames)
    #     print(f'Parsing clips from {split} set - {num_clips} clips total')

    # else:
        # * modified
        # obviously we don't need camera poses for train split
    # all_clips_json = clips_json['train']
                    #  clips_json['val']+\
                    #  clips_json['test']
    # all_clips_json = clips_json['val']+\
    #                  clips_json['test']

    clips_filenames = my_json['train'] + my_json['val'] + my_json['test']
    # [x for x in clips_filenames if\
    #                     x.split('.')[0] in all_clips_json
    #                     ]
    num_clips = len(clips_filenames)
    print(f'Parsing clips from ALL sets - {num_clips} clips total')
    # else:
    #     num_clips = len(clips_filenames)
    #     print(f'Parsing ALL clips - {num_clips} clips total')

    def frame_extractor(inputs):

        clip_filename = inputs['input']
        clip_directory = inputs['clip_dir']
        output_directory = inputs['output_dir']

        fe = FrameExtractor()

        filename = os.path.join(clip_directory, clip_filename+'.mp4')
        clip_name_uid = clip_filename.split('.')[0]
        output_dir_clip = os.path.join(output_directory, clip_name_uid)

        # skip if already processed
        # if os.path.isdir(output_dir_clip):
        #     if len(os.listdir(output_dir_clip)) > 2000:
        #         return

        Path(output_dir_clip).mkdir(parents=True, exist_ok=True)
        print("Doing ",filename)
        fe.extract(filename, output_dir_clip)

    inputs = [{
        'input': x,
        'clip_dir': clip_dir,
        'output_dir': output_dir
    } for x in clips_filenames]

    pool = Pool(args.j)

    _ = list(
        tqdm(pool.imap_unordered(frame_extractor, inputs), total=len(inputs)))
