import 'package:flutter/material.dart';
import 'voice_chat_button.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Voice Chat')),
        body: const Center(child: VoiceChatButton()),
      ),
    );
  }
}
