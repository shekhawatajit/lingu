import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:http_parser/http_parser.dart' show MediaType;

/// Change this to your backend. Examples:
/// - Android emulator: http://10.0.2.2:8000
/// - iOS simulator:    http://127.0.0.1:8000
/// - Real device:      http://<your-LAN-IP>:8000
const String kServerBase = 'http://localhost:8000';

class VoiceChatButton extends StatefulWidget {
  const VoiceChatButton({super.key});

  @override
  State<VoiceChatButton> createState() => _VoiceChatButtonState();
}

class _VoiceChatButtonState extends State<VoiceChatButton> {
  final _rec = AudioRecorder();
  final _player = AudioPlayer();
  bool _recording = false;
  bool _thinking = false;
  String? _currentPath;

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  Future<bool> _ensureMicPermission() async {
    if (kIsWeb) return true; // browser prompts itself
    var status = await Permission.microphone.status;
    if (!status.isGranted) {
      status = await Permission.microphone.request();
    }
    return status.isGranted;
  }

  Future<void> _start() async {
    if (!await _ensureMicPermission()) return;
    final canRecord = await _rec.hasPermission();
    if (!canRecord) return;

    setState(() => _recording = true);

    // Pick a temp path (mobile). On web we omit explicit path.
    String? outPath;
    if (!kIsWeb) {
      final dir = await getTemporaryDirectory();
      final fileName = 'utterance_${DateTime.now().millisecondsSinceEpoch}';
      // Choose a common container; server will transcode anyway.
      outPath = '${dir.path}/$fileName.m4a';
    }

    await _rec.start(
      RecordConfig(
        encoder: AudioEncoder.aacLc,
        sampleRate: 24000,
        numChannels: 1,
        bitRate: 64000,
      ),
      // path is required; on web it's ignored but must be provided
      path: outPath ?? 'web_record.m4a',
    );

    _currentPath = outPath; // null on web
  }

  Future<void> _stopAndSend() async {
    if (!_recording) return;
    setState(() => _recording = false);

    // On mobile, stop() returns a file path. On web, it returns a blob URL; use getFile.
    final stoppedPath = await _rec.stop();
    String? path = stoppedPath;
    Uint8List? webBytes;

    try {
      setState(() => _thinking = true);

      http.MultipartRequest req = http.MultipartRequest(
        'POST',
        Uri.parse('$kServerBase/s2s'),
      );

      if (kIsWeb) {
        // record_web: fetch blob from URL, turn to bytes
        final uri = Uri.parse(stoppedPath!);
        final resp = await http.get(uri);
        webBytes = resp.bodyBytes;
        req.files.add(
          http.MultipartFile.fromBytes(
            'file',
            webBytes,
            filename: 'speech.webm',
            contentType: MediaType('audio', 'webm'),
          ),
        );
      } else {
        path ??= _currentPath;
        if (path == null) throw Exception('Recording file path not found.');
        req.files.add(
          await http.MultipartFile.fromPath(
            'file',
            path,
            contentType: MediaType(
              'audio',
              'mp4',
            ), // m4a=aac; server ignores exact mime
          ),
        );
      }

      final streamed = await req.send();
      if (streamed.statusCode != 200) {
        final body = await streamed.stream.bytesToString();
        throw Exception('S2S failed: ${streamed.statusCode} $body');
      }
      final bytes = await streamed.stream.toBytes();

      // Save reply WAV to a temp file, then play
      final tmpDir = await getTemporaryDirectory();
      final replyPath =
          '${tmpDir.path}/reply_${DateTime.now().millisecondsSinceEpoch}.wav';
      final f = File(replyPath);
      await f.writeAsBytes(bytes, flush: true);

      await _player.setFilePath(replyPath);
      _player.play();
    } catch (e) {
      debugPrint('S2S error: $e');
      if (context.mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Voice error: $e')));
      }
    } finally {
      if (mounted) setState(() => _thinking = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        GestureDetector(
          onTapDown: (_) => _start(), // start immediately on press
          onTapUp: (_) => _stopAndSend(), // stop on release
          onTapCancel: () => _stopAndSend(), // also stop if finger slides off
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 28),
            decoration: BoxDecoration(
              color: _recording
                  ? const Color(0xFFEF4444)
                  : const Color(0xFF0EA5E9),
              borderRadius: BorderRadius.circular(999),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.15),
                  blurRadius: 10,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: Text(
              _recording ? 'Listening… release to send' : 'Hold to talk',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        if (_thinking) const Text('Thinking…'),
      ],
    );
  }
}
