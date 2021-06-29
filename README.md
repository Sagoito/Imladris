# Imladris
This university group project aims to develop a neural network architecture that allows the removal of objects from photos added by the user using deep learning techniques.

# Members of group

<table>
  <tr>
    <td align="center"><a href="https://github.com/Delci0r"><img src="https://avatars.githubusercontent.com/u/51274280?v=4" width="100px;" alt=""/><br /><sub><b>Kacper Derlatka</b></sub></a><br /><a href="https://github.com/Delcior">Github</a></td>
    <td align="center"><a href="https://github.com/Sagoito"><img src="https://avatars.githubusercontent.com/u/22767772?v=4" width="100px;" alt=""/><br /><sub><b>Damian Książek</b></sub></a><br /><a href="https://github.com/Sagoito">Github</a> </td>
    <td align="center"><a href="https://github.com/LukaszMazurek"><img src="https://avatars.githubusercontent.com/u/64841128?v=4" width="100px;" alt=""/><br /><sub><b>Łukasz Mazurek</b></sub></a><br /><a href="https://github.com/LukaszMazurek">Github</a></td>
    <td align="center"><a href="https://github.com/OpaluSunflower"><img src="https://avatars.githubusercontent.com/u/69316229?v=4" width="100px;" alt=""/><br /><sub><b>Bartłomiej Guz</b></sub></a><br /><a href="https://github.com/OpaluSunflower">Github</a> </td>
  </tr>
</table>

# Jira links
Damian Książek
https://imladris.atlassian.net/issues/?filter=10000&jql=project%20%3D%20IR%20AND%20issuetype%20in%20(Epic%2C%20Task)%20AND%20reporter%20in%20(5f8ab925b2964c006e89552f)%20ORDER%20BY%20Rank%20ASC

Kacper Derlatka
https://imladris.atlassian.net/issues/?filter=10000&jql=project%20%3D%20IR%20AND%20issuetype%20in%20(Epic%2C%20Task)%20AND%20reporter%20in%20(5f8abce769323300767318be)%20ORDER%20BY%20Rank%20ASC

Łukasz Mazurek
https://imladris.atlassian.net/issues/?filter=10000&jql=project%20%3D%20IR%20AND%20issuetype%20in%20(Epic%2C%20Task)%20AND%20reporter%20in%20(5f8abce90c96130069c426da)%20ORDER%20BY%20Rank%20ASC

Bartłomiej Guz
https://imladris.atlassian.net/issues/?filter=10000&jql=project%20%3D%20IR%20AND%20issuetype%20in%20(Epic%2C%20Task)%20AND%20reporter%20in%20(5f8abce856eec000768e9f17)%20ORDER%20BY%20Rank%20ASC


# Setup 
All you need to do is
<ul>
<li>Download weights mask_rcnn_coco.h5 (https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) and put it in src folder</li>
<li>Download weights for inpaiting network https://drive.google.com/drive/folders/1CUQa26Pb_AioJjBYTpX3JkPme3M4WcJZ?hl=pl and put it in generative_inpaiting/model_logs/release_places2_256 folder</li>
<li>Download http://images.cocodataset.org/annotations/annotations_trainval2017.zip and put it in src/annotations/ folder</li>
<li>Also you need to use pip install -r re quirements.txt to install other requirements</li>
