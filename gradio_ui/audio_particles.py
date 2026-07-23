"""
Audio Particles Visualization for Karaoke Videos
Creates stunning particle effects synchronized with music
"""

import numpy as np
import cv2
import librosa
import logging
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import colorsys
import random
import math

logger = logging.getLogger(__name__)

class AudioParticleVisualizer:
    def __init__(self, audio_path, resolution="1080p", fps=30, use_beats=True):
        self.audio_path = audio_path
        self.fps = fps
        self.use_beats = use_beats
        
        # Resolution settings
        res_map = {
            "360p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160)
        }
        self.width, self.height = res_map.get(resolution, (1920, 1080))
        
        # Load and analyze audio
        logger.info("Loading audio for particle visualization...")
        self.y, self.sr = librosa.load(audio_path)
        self.duration = len(self.y) / self.sr
        
        # Extract audio features
        self._analyze_audio()
        
        # Particle system
        self.particles = []
        self.max_particles = 260

        # Visual zones for varied effects
        self.zones = {
            'left': {'x': self.width * 0.2, 'y': self.height * 0.5},
            'right': {'x': self.width * 0.8, 'y': self.height * 0.5},
            'top': {'x': self.width * 0.5, 'y': self.height * 0.2},
            'bottom': {'x': self.width * 0.5, 'y': self.height * 0.8},
        }

        # Visual style state
        self.scale = min(self.width, self.height) / 1080.0
        self.center = (self.width // 2, self.height // 2)
        self.shockwaves = []
        self._spec_display = None   # spectrum ring values with punchy attack/decay
        self._current_hue = 0.55
        self._beat_pulse = 0.0

        # Precomputed radial falloff for the pulsing vignette background
        yy, xx = np.mgrid[0:self.height, 0:self.width].astype(np.float32)
        cx, cy = self.width / 2.0, self.height / 2.0
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        dist /= dist.max()
        self.vignette = (1.0 - dist) ** 2
        
    def _analyze_audio(self):
        """Extract various audio features for visualization"""
        logger.info("Analyzing audio features...")
        
        # Beat tracking with stricter parameters
        self.tempo, self.beats = librosa.beat.beat_track(
            y=self.y, 
            sr=self.sr,
            trim=False
        )
        self.beat_times = librosa.frames_to_time(self.beats, sr=self.sr)
        
        # Onset detection
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.onsets = librosa.onset.onset_detect(
            onset_envelope=self.onset_env, 
            sr=self.sr, 
            units='time',
            backtrack=True
        )
        
        # RMS energy
        self.rms = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        
        # Spectral features
        self.spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)[0]
        self.spectral_rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)[0]
        
        # Percussive and harmonic separation
        self.y_harmonic, self.y_percussive = librosa.effects.hpss(self.y)
        
        # Get percussive strength for better beat sync
        self.percussive_rms = librosa.feature.rms(y=self.y_percussive, frame_length=2048, hop_length=512)[0]
        
        # Create interpolation functions
        time_frames = np.linspace(0, self.duration, len(self.rms))
        self.rms_interp = interp1d(time_frames, self.rms, kind='cubic', fill_value='extrapolate')
        self.centroid_interp = interp1d(time_frames, self.spectral_centroids, kind='cubic', fill_value='extrapolate')
        self.percussive_interp = interp1d(time_frames, self.percussive_rms, kind='cubic', fill_value='extrapolate')
        
        # Smooth the RMS
        self.rms_smooth = gaussian_filter1d(self.rms, sigma=2)
        self.rms_smooth_interp = interp1d(time_frames, self.rms_smooth, kind='cubic', fill_value='extrapolate')

        # Mel spectrogram for the radial spectrum ring
        mel = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=64, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        self.mel_spec = np.clip((mel_db + 60) / 60, 0, 1).astype(np.float32)
        self.mel_times = np.linspace(0, self.duration, self.mel_spec.shape[1])

        # Bass energy (low mel bands) drives the vignette pulse and ring radius
        bass = self.mel_spec[:8].mean(axis=0)
        bass = gaussian_filter1d(bass, sigma=2)
        bass = (bass - bass.min()) / (bass.max() - bass.min() + 1e-6)
        self.bass_interp = interp1d(self.mel_times, bass, kind='linear', fill_value='extrapolate')

        logger.info(f"Found {len(self.beat_times)} beats, {len(self.onsets)} onsets")
        logger.info(f"Tempo: {float(self.tempo):.1f} BPM")
        
    def _palette_hue(self, t):
        """Slowly drifting base hue so the whole scene stays color-coherent."""
        return (0.55 + 0.10 * math.sin(t * 0.15) + t * 0.008) % 1.0

    def _hsv_color(self, h, s=1.0, v=1.0):
        rgb = colorsys.hsv_to_rgb(h % 1.0, s, v)
        return [int(c * 255) for c in rgb]

    def create_particle(self, x, y, particle_type="normal"):
        """Create a new particle with physics properties"""
        particle = {
            'x': x,
            'y': y,
            'vx': 0,
            'vy': 0,
            'life': 1.0,
            'size': random.uniform(3, 8),
            'original_size': random.uniform(3, 8),
            'type': particle_type,
            'color': [255, 255, 255],
            'trail': [],
            'age': 0,
            'rotation': random.uniform(0, 2 * np.pi),
            'rotation_speed': random.uniform(-0.1, 0.1)
        }
        
        if particle_type == "beat_explosion":
            # Explosive particles from beat
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(15, 25)
            particle['vx'] = np.cos(angle) * speed
            particle['vy'] = np.sin(angle) * speed
            particle['size'] = random.uniform(10, 20)
            particle['original_size'] = particle['size']
            # Complementary hue so beat hits pop against the scene
            particle['color'] = self._hsv_color(self._current_hue + 0.5 + random.uniform(-0.05, 0.05), 0.9, 1.0)
            particle['life'] = 1.5
            
        elif particle_type == "wave":
            # Particles that move in waves
            particle['vx'] = random.uniform(-3, 3)
            particle['vy'] = random.uniform(-2, 2)
            particle['size'] = random.uniform(4, 10)
            particle['original_size'] = particle['size']
            particle['color'] = self._hsv_color(self._current_hue + random.uniform(-0.06, 0.06), 0.8, 1.0)
            particle['wave_offset'] = random.uniform(0, 2 * np.pi)
            
        elif particle_type == "spiral":
            # Spiral moving particles
            particle['spiral_angle'] = random.uniform(0, 2 * np.pi)
            particle['spiral_radius'] = 0
            particle['spiral_speed'] = random.uniform(0.05, 0.1)
            particle['size'] = random.uniform(5, 12)
            particle['original_size'] = particle['size']
            particle['color'] = self._hsv_color(self._current_hue + 0.15, 0.85, 1.0)
            
        elif particle_type == "fountain":
            # Fountain effect from bottom
            particle['vx'] = random.uniform(-2, 2)
            particle['vy'] = random.uniform(-15, -10)
            particle['size'] = random.uniform(3, 8)
            particle['original_size'] = particle['size']
            particle['color'] = self._hsv_color(self._current_hue - 0.15, 0.75, 1.0)
            
        return particle
    
    def update_particles(self, current_time):
        """Update particle positions and properties"""
        # Get current audio features
        rms_val = float(self.rms_smooth_interp(current_time))
        centroid_val = float(self.centroid_interp(current_time))
        percussive_val = float(self.percussive_interp(current_time))
        
        # Normalize values
        rms_normalized = np.clip(rms_val * 8, 0, 1)
        centroid_normalized = np.clip(centroid_val / 4000, 0, 1)
        percussive_normalized = np.clip(percussive_val * 10, 0, 1)
        
        # Beat detection with strength
        is_beat = False
        beat_strength = 0
        if self.use_beats:
            beat_distances = [abs(current_time - bt) for bt in self.beat_times]
            if beat_distances:
                min_distance = min(beat_distances)
                if min_distance < 0.1:
                    is_beat = True
                    beat_strength = 1.0 - (min_distance / 0.1)
        self._beat_pulse = beat_strength if is_beat else 0.0

        # Spawn an expanding shockwave ring on strong beats
        if is_beat and beat_strength > 0.5:
            if not self.shockwaves or self.shockwaves[-1]['radius'] > 80 * self.scale:
                self.shockwaves.append({
                    'radius': 40 * self.scale,
                    'life': 1.0,
                    'speed': (18 + 14 * beat_strength) * self.scale,
                    'color': self._hsv_color(self._current_hue + 0.5, 0.6, 1.0),
                })
        
        # Onset detection
        onset_threshold = 0.05
        is_onset = any(abs(current_time - ot) < onset_threshold for ot in self.onsets)
        
        # Create particles based on audio events
        if is_beat and beat_strength > 0.3:
            # Create beat explosions from multiple points
            explosion_points = [
                (self.width // 2, self.height // 2),  # Center
                (self.width // 4, self.height // 3),  # Top left
                (3 * self.width // 4, self.height // 3),  # Top right
                (self.width // 4, 2 * self.height // 3),  # Bottom left
                (3 * self.width // 4, 2 * self.height // 3),  # Bottom right
            ]
            
            # Pick 1-3 random explosion points based on beat strength
            num_explosions = min(3, int(1 + beat_strength * 2))
            selected_points = random.sample(explosion_points, num_explosions)
            
            for point in selected_points:
                num_particles = int(10 * beat_strength * percussive_normalized)
                for _ in range(num_particles):
                    x = point[0] + random.randint(-30, 30)
                    y = point[1] + random.randint(-30, 30)
                    self.particles.append(self.create_particle(x, y, "beat_explosion"))
        
        # Wave particles on onsets
        if is_onset and not is_beat:
            # Create wave from sides
            side = random.choice(['left', 'right'])
            num_particles = int(5 + rms_normalized * 10)
            for i in range(num_particles):
                if side == 'left':
                    x = 0
                    vx = random.uniform(5, 10)
                else:
                    x = self.width
                    vx = random.uniform(-10, -5)
                    
                y = self.height // 2 + random.randint(-self.height // 4, self.height // 4)
                particle = self.create_particle(x, y, "wave")
                particle['vx'] = vx
                self.particles.append(particle)
        
        # Continuous effects based on RMS
        if random.random() < rms_normalized * 0.5:
            # Fountain effect from bottom
            if random.random() < 0.3:
                num_particles = int(3 + rms_normalized * 5)
                for _ in range(num_particles):
                    x = random.randint(self.width // 3, 2 * self.width // 3)
                    y = self.height
                    self.particles.append(self.create_particle(x, y, "fountain"))
            
            # Spiral particles
            if random.random() < 0.3:
                zone = random.choice(list(self.zones.values()))
                particle = self.create_particle(zone['x'], zone['y'], "spiral")
                self.particles.append(particle)
        
        # Update existing particles
        for particle in self.particles:
            particle['age'] += 1
            
            # Type-specific movement
            if particle['type'] == "beat_explosion":
                # Explosive movement with drag
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['vx'] *= 0.9
                particle['vy'] *= 0.9
                particle['vy'] += 0.3  # Gravity
                
            elif particle['type'] == "wave":
                # Wave motion
                wave_offset = particle.get('wave_offset', 0)
                particle['x'] += particle['vx']
                particle['y'] += particle['vy'] + np.sin(particle['age'] * 0.2 + wave_offset) * 2
                particle['vx'] *= 0.98
                
            elif particle['type'] == "spiral":
                # Spiral motion
                particle['spiral_angle'] += particle['spiral_speed']
                particle['spiral_radius'] += 2
                center_x = particle.get('start_x', particle['x'])
                center_y = particle.get('start_y', particle['y'])
                particle['x'] = center_x + np.cos(particle['spiral_angle']) * particle['spiral_radius']
                particle['y'] = center_y + np.sin(particle['spiral_angle']) * particle['spiral_radius']
                
            elif particle['type'] == "fountain":
                # Fountain physics
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['vy'] += 0.5  # Gravity
                particle['vx'] *= 0.95
                
            else:
                # Normal particles
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['vx'] *= 0.95
                particle['vy'] *= 0.95
            
            # Rotation
            particle['rotation'] += particle['rotation_speed']
            
            # Update trail
            particle['trail'].append((int(particle['x']), int(particle['y'])))
            max_trail = 10 if particle['type'] == "beat_explosion" else 6
            if len(particle['trail']) > max_trail:
                particle['trail'].pop(0)
            
            # Dynamic color based on audio, anchored to the drifting palette
            if particle['type'] in ["wave", "normal"]:
                hue = self._current_hue + centroid_normalized * 0.15
                saturation = 0.6 + rms_normalized * 0.4
                value = 0.8 + percussive_normalized * 0.2
                particle['color'] = self._hsv_color(hue, saturation, value)
            
            # Size pulsing
            if particle['type'] != "beat_explosion":
                base_size = particle['original_size']
                particle['size'] = base_size * (0.8 + rms_normalized * 0.4)
            
            # Fade out
            fade_rates = {
                "beat_explosion": 0.02,
                "wave": 0.015,
                "spiral": 0.012,
                "fountain": 0.018,
                "normal": 0.02
            }
            particle['life'] -= fade_rates.get(particle['type'], 0.02)
            
        # Remove dead particles
        self.particles = [p for p in self.particles if p['life'] > 0 and 
                         -100 < p['x'] < self.width + 100 and 
                         -100 < p['y'] < self.height + 100]
        
        # Limit particle count
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]
    
    def _draw_background(self, bass):
        """Bass-pulsing radial vignette in the current palette hue."""
        color = np.array(self._hsv_color(self._current_hue, 0.9, 1.0), dtype=np.float32)
        brightness = 0.05 + 0.13 * bass
        return (self.vignette[:, :, None] * (color * brightness)[None, None, :]).astype(np.uint8)

    def _draw_spectrum_ring(self, fg, current_time, bass):
        """Slowly rotating, mirrored circular spectrum analyzer around the center."""
        idx = min(int(current_time / self.duration * (self.mel_spec.shape[1] - 1)),
                  self.mel_spec.shape[1] - 1)
        spec = gaussian_filter1d(self.mel_spec[:, idx], sigma=1.2)

        # Punchy attack, smooth decay
        if self._spec_display is None:
            self._spec_display = spec
        else:
            self._spec_display = np.maximum(spec, self._spec_display * 0.88)
        disp = self._spec_display

        n_bars = 96
        half = n_bars // 2
        cx, cy = self.center
        base_r = min(self.width, self.height) * (0.16 + 0.035 * bass)
        max_len = min(self.width, self.height) * 0.16
        rot = current_time * 0.25
        thickness = max(2, int(3 * self.scale))

        for i in range(n_bars):
            # Mirror the spectrum so the ring is left/right symmetric
            j = i if i < half else n_bars - 1 - i
            band = float(disp[int(j / half * (len(disp) - 1))])
            angle = 2 * np.pi * i / n_bars - np.pi / 2 + rot
            length = (0.12 + band ** 1.3) * max_len
            x0 = int(cx + math.cos(angle) * base_r)
            y0 = int(cy + math.sin(angle) * base_r)
            x1 = int(cx + math.cos(angle) * (base_r + length))
            y1 = int(cy + math.sin(angle) * (base_r + length))
            color = self._hsv_color(self._current_hue + 0.25 * (j / half), 0.85, 0.6 + 0.4 * band)
            cv2.line(fg, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

        # Inner ring that breathes with the bass
        cv2.circle(fg, (cx, cy), max(1, int(base_r * 0.97)),
                   self._hsv_color(self._current_hue, 0.5, 0.9),
                   max(2, int(2 * self.scale)), cv2.LINE_AA)

    def _draw_shockwaves(self, fg):
        """Expanding rings launched on strong beats."""
        for wave in self.shockwaves:
            wave['radius'] += wave['speed']
            wave['life'] -= 0.04
        self.shockwaves = [w for w in self.shockwaves if w['life'] > 0]

        for wave in self.shockwaves:
            color = [int(c * wave['life']) for c in wave['color']]
            radius = int(wave['radius'])
            cv2.circle(fg, self.center, radius, color,
                       max(2, int(6 * self.scale * wave['life'])), cv2.LINE_AA)
            cv2.circle(fg, self.center, int(radius * 0.92),
                       [int(c * 0.5) for c in color],
                       max(1, int(2 * self.scale)), cv2.LINE_AA)

    def draw_frame(self, current_time):
        """Draw a single frame of the visualization"""
        self._current_hue = self._palette_hue(current_time)
        bass = float(np.clip(self.bass_interp(current_time), 0, 1))

        # Layer 1: pulsing vignette background
        bg = self._draw_background(bass)

        # Layer 2: everything luminous is drawn on a black fg buffer,
        # which later gets a bloom pass for the neon glow
        fg = np.zeros_like(bg)

        self.update_particles(current_time)
        self._draw_spectrum_ring(fg, current_time, bass)
        self._draw_shockwaves(fg)

        # Draw particles with different effects based on type
        for particle in self.particles:
            x, y = int(particle['x']), int(particle['y'])
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            
            # Draw trail
            if len(particle['trail']) > 1 and particle['life'] > 0.1:
                for i in range(1, len(particle['trail'])):
                    alpha = (i / len(particle['trail'])) * particle['life'] * 0.6
                    thickness = max(1, int(particle['size'] * alpha * 0.3))
                    color = [int(c * alpha) for c in particle['color']]
                    cv2.line(fg, particle['trail'][i-1], particle['trail'][i], color, thickness)
            
            size = int(particle['size'] * particle['life'])
            if size > 0:
                # Special effects for different particle types
                if particle['type'] == "beat_explosion":
                    # Double circle effect
                    cv2.circle(fg,(x, y), int(size * 1.5), 
                              [int(c * particle['life'] * 0.5) for c in particle['color']], 
                              2, cv2.LINE_AA)
                    cv2.circle(fg,(x, y), size, 
                              [int(c * particle['life']) for c in particle['color']], 
                              -1, cv2.LINE_AA)
                elif particle['type'] == "spiral":
                    # Star shape for spiral particles
                    points = []
                    for i in range(5):
                        angle = particle['rotation'] + i * 2 * np.pi / 5
                        px = x + int(size * np.cos(angle))
                        py = y + int(size * np.sin(angle))
                        points.append([px, py])
                    points = np.array(points, np.int32)
                    cv2.fillPoly(fg, [points],
                                [int(c * particle['life']) for c in particle['color']])
                else:
                    # Regular circle
                    cv2.circle(fg,(x, y), size, 
                              [int(c * particle['life']) for c in particle['color']], 
                              -1, cv2.LINE_AA)
                
                # Add bright core to all particles
                core_size = max(1, size // 3)
                brightness = min(255, int(255 * particle['life'] * 1.2))
                cv2.circle(fg,(x, y), core_size, [brightness, brightness, brightness], -1)
        
        # Bloom pass: a blurred copy of the luminous layer added back on top
        # gives everything a neon glow
        sigma = max(4.0, 8.0 * self.scale)
        glow = cv2.GaussianBlur(fg, (0, 0), sigma)
        frame = cv2.add(bg, fg)
        frame = cv2.addWeighted(frame, 1.0, glow, 0.8, 0)

        # Beat punch: quick zoom-in on strong beats for physical impact
        if self._beat_pulse > 0.4:
            zoom = 1.0 + 0.035 * self._beat_pulse
            new_w, new_h = int(self.width * zoom), int(self.height * zoom)
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            x0 = (new_w - self.width) // 2
            y0 = (new_h - self.height) // 2
            frame = scaled[y0:y0 + self.height, x0:x0 + self.width]

        return frame
    
    def generate_video(self, output_path, progress_callback=None):
        """Generate the complete particle visualization video"""
        logger.info(f"Generating particle visualization video: {output_path}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        total_frames = int(self.duration * self.fps)
        logger.info(f"Total frames to generate: {total_frames}")
        
        # Process every other frame for speed
        frame_skip = 2
        
        for frame_idx in range(0, total_frames, frame_skip):
            current_time = frame_idx / self.fps
            
            # Generate frame
            frame = self.draw_frame(current_time)
            
            # Write frame multiple times to maintain fps
            for _ in range(frame_skip):
                out.write(frame)
            
            # Progress callback
            if progress_callback and frame_idx % 60 == 0:
                progress = frame_idx / total_frames
                progress_callback(progress, f"Generating particles... {int(progress * 100)}%")
            
            # Log progress
            if frame_idx % (30 * self.fps) == 0:
                logger.info(f"Progress: {frame_idx}/{total_frames} frames ({int(frame_idx/total_frames*100)}%)")
        
        out.release()
        logger.info("Particle visualization complete!")
        return output_path


def integrate_particle_visualizer(audio_filepath, output_path, resolution="1080p", progress_callback=None):
    """Integration function for your existing code"""
    try:
        visualizer = AudioParticleVisualizer(audio_filepath, resolution=resolution)
        video_path = visualizer.generate_video(output_path, progress_callback)
        return video_path
    except Exception as e:
        logger.error(f"Error creating particle visualization: {e}")
        return None