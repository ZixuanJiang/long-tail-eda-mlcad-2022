## baseline
python train_batch.py --reweighting CrossEntropy --model_save_path ./checkpoints/CE.pth > ./logs/CE.log

### reweighting
## cost sensitive CE, gamma = 1.0
python train_batch.py --reweighting CostSensitiveCE --model_save_path ./checkpoints/CSCE_1.pth > ./logs/CSCE_1.log
## focal
python train_batch.py --reweighting FocalLoss --model_save_path ./checkpoints/focal.pth > ./logs/focal.log
## class balance focal
python train_batch.py --reweighting ClassBalanceFocal --model_save_path ./checkpoints/CBfocal.pth > ./logs/CBfocal.log
## class balance CE
python train_batch.py --reweighting ClassBalanceCE --model_save_path ./checkpoints/CBCE.pth > ./logs/CBCE.log
## CE label smooth
python train_batch.py --reweighting CrossEntropyLabelSmooth --model_save_path ./checkpoints/CELS.pth > ./logs/CELS.log
## CE label aware smooth
python train_batch.py --reweighting CrossEntropyLabelAwareSmooth --model_save_path ./checkpoints/CELAS.pth > ./logs/CELAS.log
## LDAM
python train_batch.py --reweighting LDAMLoss --model_save_path ./checkpoints/LDAM.pth > ./logs/LDAM.log
## CDT
python train_batch.py --reweighting CDT --model_save_path ./checkpoints/CDT.pth > ./logs/CDT.log
## Balanced Softmax CE
python train_batch.py --reweighting BalancedSoftmaxCE --model_save_path ./checkpoints/BSCE.pth > ./logs/BSCE.log

### resampling
## balance
python train_batch.py --resampling balance --model_save_path ./checkpoints/balance.pth > ./logs/balance.log
## square
python train_batch.py --resampling square --model_save_path ./checkpoints/square.pth > ./logs/square.log
## progressive
python train_batch.py --resampling progressive --model_save_path ./checkpoints/progressive.pth > ./logs/progressive.log

### two_stage reweighting
## cost sensitive CE, gamma = 1.0
python train_batch.py --two_stage drw --reweighting CostSensitiveCE --model_save_path ./checkpoints/DS_CSCE_1.pth > ./logs/DS_CSCE_1.log
## focal
python train_batch.py --two_stage drw --reweighting FocalLoss --model_save_path ./checkpoints/DS_focal.pth > ./logs/DS_focal.log
## class balance focal
python train_batch.py --two_stage drw --reweighting ClassBalanceFocal --model_save_path ./checkpoints/DS_CBfocal.pth > ./logs/DS_CBfocal.log
## class balance CE
python train_batch.py --two_stage drw --reweighting ClassBalanceCE --model_save_path ./checkpoints/DS_CBCE.pth > ./logs/DS_CBCE.log
## CE label smooth
python train_batch.py --two_stage drw --reweighting CrossEntropyLabelSmooth --model_save_path ./checkpoints/DS_CELS.pth > ./logs/DS_CELS.log
## CE label aware smooth
python train_batch.py --two_stage drw --reweighting CrossEntropyLabelAwareSmooth --model_save_path ./checkpoints/DS_CELAS.pth > ./logs/DS_CELAS.log
## LDAM
python train_batch.py --two_stage drw --reweighting LDAMLoss --model_save_path ./checkpoints/DS_LDAM.pth > ./logs/DS_LDAM.log
## CDT
python train_batch.py --two_stage drw --reweighting CDT --model_save_path ./checkpoints/DS_CDT.pth > ./logs/DS_CDT.log
## Balanced Softmax CE
python train_batch.py --two_stage drw --reweighting BalancedSoftmaxCE --model_save_path ./checkpoints/DS_BSCE.pth > ./logs/DS_BSCE.log

## two_stage resampling
# balance
python train_batch.py --two_stage drs --resampling balance --model_save_path ./checkpoints/DS_balance.pth > ./logs/DS_balance.log
# square
python train_batch.py --two_stage drs --resampling square --model_save_path ./checkpoints/DS_square.pth > ./logs/DS_square.log
# progressive
python train_batch.py --two_stage drs --resampling progressive --model_save_path ./checkpoints/DS_square.pth > ./logs/DS_progressive.log


