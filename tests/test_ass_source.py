"""Original-media metadata and ASS canvas regression tests (no models/network)."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.temporary.real_chain_runner import media
from modules.temporary.real_chain_finish import legacy, operations

ROOT = Path(__file__).resolve().parents[1]
ASS = b'\xef\xbb\xbf[Script Info]\r\nPlayResX: 640\r\nPlayResY: 480\r\n[V4+ Styles]\r\nStyle: unchanged\r\n[Events]\r\nDialogue: unchanged\r\n'


class SourceMetadataTest(unittest.TestCase):
    def probe(self, streams, duration='10.5'):
        result = SimpleNamespace(stdout=json.dumps({'format': {'duration': duration}, 'streams': streams}))
        with patch.object(media.subprocess, 'run', return_value=result) as run:
            data = media.source_metadata(ROOT / 'video.mp4', 'cuda')
        self.assertEqual(run.call_count, 1)
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertEqual(data['device'], 'cuda')
        return data

    def test_source_dimensions(self):
        data = self.probe([{'codec_type': 'video', 'width': 1730, 'height': 1078}])
        self.assertEqual((data['video']['width'], data['video']['height']), (1730, 1078))
        self.assertEqual(data['duration_s'], 10.5)
        self.assertTrue(Path(data['media_path']).is_absolute())

    def test_rotation_and_non_square_pixels(self):
        data = self.probe([{'codec_type': 'video', 'width': 720, 'height': 576,
                            'sample_aspect_ratio': '16:15', 'side_data_list': [{'rotation': -90}]}])
        self.assertEqual((data['video']['width'], data['video']['height']), (576, 768))

    def test_audio_and_attached_artwork(self):
        data = self.probe([{'codec_type': 'audio'}, {'codec_type': 'video', 'disposition': {'attached_pic': 1}}])
        self.assertIsNone(data['video'])

    def test_missing_duration_and_sar(self):
        for duration in (None, 'N/A', 'nan', '-1'):
            data = self.probe([{'codec_type': 'video', 'width': 10, 'height': 20,
                                'sample_aspect_ratio': None}], duration)
            self.assertIsNone(data['duration_s'])
            self.assertEqual(data['video']['width'], 10)

    def test_reject_invalid_geometry(self):
        for extra in ({'width': 0}, {'tags': {'rotate': '45'}}):
            with self.assertRaises(ValueError):
                self.probe([{'codec_type': 'video', 'width': 10, 'height': 20, **extra}])


class CanvasTest(unittest.TestCase):
    def test_only_canvas_bytes_change_in_both_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'export').mkdir()
            source = {'schema': 1, 'media_path': str(work / 'video.mp4'),
                      'video': {'width': 1730, 'height': 1078}}
            (work / 'media-source.json').write_text(json.dumps(source), encoding='utf-8')
            for name in ('subtitles.ass', 'export/subtitles.ass'):
                (work / name).write_bytes(ASS)
            legacy.set_ass_canvas(work, legacy.export_source(work))
            expected = ASS.replace(b'PlayResX: 640', b'PlayResX: 1730').replace(b'PlayResY: 480', b'PlayResY: 1078')
            for name in ('subtitles.ass', 'export/subtitles.ass'):
                self.assertEqual((work / name).read_bytes(), expected)
            self.assertTrue((work / 'reports/export-media.json').is_file())

    def test_audio_legacy_and_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'subtitles.ass').write_bytes(ASS)
            self.assertIsNone(legacy.export_source(work))
            for source in (None, {'video': None}):
                legacy.set_ass_canvas(work, source)
                self.assertEqual((work / 'subtitles.ass').read_bytes(), ASS)
            (work / 'media-source.json').write_text(json.dumps({'schema': 1, 'media_path': 'relative'}))
            with self.assertRaises(ValueError):
                legacy.export_source(work)

    def test_malformed_canvas_does_not_partially_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / 'export').mkdir()
            (work / 'subtitles.ass').write_bytes(ASS)
            for bad in (ASS.replace(b'PlayResY: 480', b'other: 480'),
                        ASS.replace(b'PlayResY: 480', b'PlayResY: 480\r\nPlayResY: 480')):
                (work / 'export/subtitles.ass').write_bytes(bad)
                with self.assertRaises(ValueError):
                    legacy.set_ass_canvas(work, {'video': {'width': 1730, 'height': 1078}})
                self.assertEqual((work / 'subtitles.ass').read_bytes(), ASS)

    def test_export_passes_source_and_changes_canvas_before_gate(self):
        legacy._legacy_imports()
        from qwen_asr import commands
        parent = ROOT / 'runs/experiments'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            work = Path(tmp) / 'work'
            operations.synthetic(work)
            source = {'schema': 1, 'media_path': str(work / 'source.mp4'),
                      'video': {'width': 1730, 'height': 1078}}
            (work / 'media-source.json').write_text(json.dumps(source))
            def export(args, paths):
                self.assertEqual(args.media_path, source['media_path'])
                (work / 'export/subtitles.srt').write_text('srt')
                for name in ('subtitles.ass', 'export/subtitles.ass'):
                    (work / name).write_bytes(ASS)
                return 0
            def gate(args, paths):
                self.assertIn(b'PlayResX: 1730', (work / 'export/subtitles.ass').read_bytes())
                paths.final_quality_report.write_text('{"status":"PASS"}')
                return 0
            with patch.object(commands, 'cmd_export', export), patch.object(commands, 'cmd_quality_gate', gate):
                self.assertEqual(legacy.run_stage('export', work)['returncode'], 0)


if __name__ == '__main__':
    unittest.main()
