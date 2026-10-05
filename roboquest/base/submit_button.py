"""Physical first-press completion for the cup inspection scene."""
from copy import deepcopy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

SUBMIT_XY = (.43, -.255)


class CupSubmitMixin:
    # Subclasses may rename the button bodies and move it; defaults keep the cup scene unchanged.
    submit_prefix = 'cup_submit'
    submit_xy = SUBMIT_XY

    def _add_submit_button(self, world, assets, table_z):
        SUBMIT_XY = self.submit_xy
        base = ET.SubElement(world, 'body', name=f'{self.submit_prefix}_button',
                             pos=f'{SUBMIT_XY[0]} {SUBMIT_XY[1]} {table_z}')
        for visual in (False, True):
            ET.SubElement(base, 'geom', name=f'{self.submit_prefix}_base'+('_visual' if visual else ''),
                type='box', pos='0 0 .010', size='.048 .043 .010', rgba='.12 .13 .15 1',
                group='1' if visual else '0', contype='0' if visual else '1',
                conaffinity='0' if visual else '1')
        cap = ET.SubElement(base, 'body', name=f'{self.submit_prefix}_moving_cap', pos='0 0 .032')
        ET.SubElement(cap, 'joint', name=f'{self.submit_prefix}_travel', type='slide', axis='0 0 -1',
            range='0 .009', limited='true', damping='2', stiffness='160', springref='0')
        for visual in (False, True):
            ET.SubElement(cap, 'geom', name=f'{self.submit_prefix}_cap'+('_visual' if visual else ''),
                type='cylinder', size='.036 .010', rgba='.78 .06 .04 1',
                mass='0' if visual else '.025', friction='1 .01 .001',
                group='1' if visual else '0', contype='0' if visual else '1',
                conaffinity='0' if visual else '1')
        ET.SubElement(assets, 'texture', name=f'{self.submit_prefix}_texture', type='2d', builtin='flat',
                      width='256', height='128', rgb1='.78 .06 .04')
        ET.SubElement(assets, 'material', name=f'{self.submit_prefix}_material', texture=f'{self.submit_prefix}_texture',
                      texrepeat='1 1', texuniform='false', specular='0', shininess='0')
        # Explicit UVs keep text upright on the moving cap. A tiny closed
        # backing gives the visual mesh nonzero volume for the native compiler.
        ET.SubElement(assets, 'mesh', name=f'{self.submit_prefix}_label_mesh',
            vertex='-.028 -.014 0 .028 -.014 0 .028 .014 0 -.028 .014 0 0 0 -.0001',
            face='0 1 2 0 2 3 1 0 4 2 1 4 3 2 4 0 3 4',texcoord='0 1 1 1 1 0 0 0 .5 .5')
        ET.SubElement(cap, 'geom', name=f'{self.submit_prefix}_label', type='mesh', mesh=f'{self.submit_prefix}_label_mesh',
            pos='0 0 .0101', material=f'{self.submit_prefix}_material', group='1',
            contype='0', conaffinity='0', density='0')
        return dict(position=[*SUBMIT_XY, table_z+.042], travel_threshold_m=.003,
                    completion='First physical robot press freezes the current score and ends the episode')

    def _setup_submit(self):
        from PIL import Image, ImageDraw, ImageFont
        m = self.sim.model
        self._button_gid = m.geom_name2id(f'{self.submit_prefix}_cap')
        self._button_qadr = m.get_joint_qpos_addr(f'{self.submit_prefix}_travel')
        label = Image.new('RGB', (256,128), (199,15,10))
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',44)
        ImageDraw.Draw(label).text((128,64),'SUBMIT',fill='white',font=font,anchor='mm')
        tid = mujoco.mj_name2id(m._model,mujoco.mjtObj.mjOBJ_TEXTURE,f'{self.submit_prefix}_texture')
        address = int(m.tex_adr[tid])
        # Image row zero is the top of the text (+Y under the explicit UVs).
        m.tex_data[address:address+256*128*3] = np.asarray(label).ravel()
        context = getattr(self.sim,'_render_context_offscreen',None)
        if context is not None:
            context.upload_texture(tid)

    def _reset_internal(self):
        super()._reset_internal()
        self.submission = None
        self.timed_out = False

    def _check_submit_contact(self):
        if self.submission is not None:
            return
        m,d = self.sim.model,self.sim.data
        if d.qpos[self._button_qadr] < .003:
            return
        for index,contact in enumerate(d.contact[:d.ncon]):
            a,b = int(contact.geom1),int(contact.geom2)
            if self._button_gid not in (a,b):
                continue
            other = b if a == self._button_gid else a
            if not (m.geom_id2name(other) or '').startswith(('robot','gripper')):
                continue
            normal = np.asarray(contact.frame).reshape(3,3)[0]*(1 if b == self._button_gid else -1)
            force = np.zeros(6)
            mujoco.mj_contactForce(m._model,d._data,index,force)
            if normal[2] > -.9 or force[0] <= 1e-5:
                continue
            self.submission = dict(step=int(self.timestep),score=deepcopy(self._current_score()),
                button_travel_m=float(d.qpos[self._button_qadr]),
                contact_geoms=[m.geom_id2name(a),m.geom_id2name(b)],
                contact_normal_on_button=normal.tolist(),normal_force_n_private=float(force[0]))
            return

    def _post_action(self, action):
        reward,done,info = super()._post_action(action)
        self._check_submit_contact()
        if self.submission is None and self.timestep >= self.horizon:
            self.timed_out = True
        info['submitted'] = self.submission is not None
        return reward,done or self.submission is not None or self.timed_out,info

    def evaluate_success(self,states=None):
        if self.submission is not None:
            return dict(deepcopy(self.submission['score']),submitted=True,
                        termination='physical_submit',submission_step=self.submission['step'])
        score = self._current_score(states)
        return dict(score,current_goal_satisfied=score['success'],success=False,
                    submitted=False,termination='timeout' if self.timed_out else 'awaiting_submit')

    def step(self, action):
        if self.submission is not None:
            raise RuntimeError('Episode ended at first physical Submit')
        if self.timed_out:
            raise RuntimeError('Episode ended without submission at timeout')
        return super().step(action)
