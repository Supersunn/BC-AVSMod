# BC-AVSMod

<!-- Improved compatibility of back to top link: See: https://github.com/Supersunn/BC-AVSMod/pull/73 -->
<a id="readme-top"></a>
<!--




<!-- PROJECT LOGO -->
<br />
  <h3 align="center">【CVPRF2026】Background-Compensated Audio-Visual Semantic Modulation Framework for Audio-Visual Event Localization</h3>

  <p align="center">
    <br />
    <a href="https://github.com/Supersunn/BC-AVSMod"><strong>Explore the docs »</strong></a>
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Overall Model Structure">About The Project</a>
      <ul>
        <li><a href="#Main Innovations">Built With</a></li>
      </ul>
    </li>
    
    <li>
      <a href="#Experiments">About The Project</a>
      <ul>
        <li><a href="#Comparison Experiments">Built With</a></li>
      </ul>
      <ul>
        <li><a href="#Qualitative Analysis">Built With</a></li>
      </ul>    
    </li>

    <li>
      <a href="#ABC-AVSMod">About The Project</a>  
    </li>
    

    <li><a href="#Cite<img width="2800" height="1574" alt="fig2" src="https://github.com/user-attachments/assets/7449b2d4-a340-4069-89ad-07c7ab1e3dbb" />
">License</a></li>

  </ol>
</details>



<!-- Overall Model Structure -->
## Overall Model Structure

<img width="2800" height="1574" alt="fig2" src="https://github.com/user-attachments/assets/d6240d95-c20c-46d1-8b38-cd4351ae0a97" />

In this work, we propose a Background-Compensated Audio-Visual Semantic Modulation (BC-AVSMod) framework, a novel CLIP-based paradigm for AVEL. To accurately capture AVB semantics, we design a refined caption generation pipeline: LLaVA produces segment-level visual captions, which are then distilled and summarized via LLaMA 3 to obtain compact yet descriptive background representations. To effectively transfer multi-modal alignment knowledge into the AVEL framework, we incorporate a sampling rate-controllable Adapter within the VLP encoders. This module leverages AudioCLIP’s pretrained audio representations, enabling semantically consistent feature extraction directly from raw audio signals. Furthermore, we propose a Semantic Calibration and Fusion (SCF) module that unifies video, audio, and AVB caption embeddings into a unified representation. The structured state-space sequence model is employed to capture intra-modal sequential dependencies, while the cross-modal attention mechanism facilitates fine-grained semantic interactions across modalities. The mixture of experts (MoE) strategy further filters modality-relevant information, complemented by a prompt-learning branch that enhances CLIP-based cross-modal alignment.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Main Innovations
our main innovations of this paper include:

* A novel text modality is introduced to describe AVE backgrounds, leveraging fine-grained LLM reasoning to enrich high-level semantic cues in audio-visual pairs.
* A modality-adaptive Adapter is integrated into VLP encoders to effectively transfer multi-modal knowledge from large-scale pretraining, improving encoders’ ability to model fine-grained audio-visual correspondences.
* An SCF module is proposed to capture intra- and inter-modal semantics via a structured state-space model and a cross-modal semantic modulation mechanism. This integrates video, audio, and AVB captions into a unified representation, further enhanced by a prompt-learning branch built atop a CLIP-based architecture.
* Extensive experiments on the AVE dataset demonstrate that BC-AVSMod achieves state-of-the-art (SOTA) performance, significantly outperforming other methods.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Experiments -->
## Experiments

### Comparison Experiments
Comparison with state-of-the-art models on AVE dataset. snow and fire denote frozen and trainable parameters, respectively. $*$ indicates results re-implemented by us. Best results are in bold, second best are underlined.

<img width="1636" height="875" alt="image" src="https://github.com/user-attachments/assets/d52b13c1-50bd-4f67-bc10-2d0b64af2952" />

### Qualitative Analysis
Visualization and qualitative analysis on the AVE dataset under the fully supervised setting. The first two rows of each example show the audio and video tracks of the same sample, while the third and fourth rows present the focus regions of CACE-Net and our model, visualized as heatmaps.
 
<img width="1814" height="973" alt="image" src="https://github.com/user-attachments/assets/c9f00b98-03a8-4faf-96d6-f3483ca28b02" />


<!-- ABC-AVSMod -->
## ABC-AVSMod
<br />
  <h3 align="center">We have optimized the Adapter and SCF modules and updated the model. Our new model ABC-AVSMod and the code will be available soon！</h3>
</div>

<!-- LICENSE -->
## Cite

```
Chao Sun, Junbo Zhang, Chuanbo Zhu, Mingjun Huang, Bo Du*. Background-Compensated Audio-Visual Semantic Modulation Framework for Audio-Visual Event Localization. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2026:7272-7281.

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
