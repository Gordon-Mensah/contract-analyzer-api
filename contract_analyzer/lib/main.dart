import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() => runApp(ContractAnalyzerApp());

class ContractAnalyzerApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Contract Analyzer',
      theme: ThemeData(primarySwatch: Colors.indigo),
      home: UploadScreen(),
    );
  }
}

class UploadScreen extends StatefulWidget {
  @override
  _UploadScreenState createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  Map<String, dynamic>? analysisResult;
  String errorMessage = '';
  bool isLoading = false;

  Future<void> analyzeContract() async {
    final picked = await FilePicker.platform.pickFiles();
    if (picked == null) return;

    final file = picked.files.first;
    final apiUrl = "https://contract-analyzer-api-ufao.onrender.com/analyze";

    setState(() {
      isLoading = true;
      analysisResult = null;
      errorMessage = '';
    });

    try {
      final request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.fields['contract_type'] = 'nda'; // Required by backend
      request.files.add(http.MultipartFile.fromBytes('file', file.bytes!, filename: file.name));

      final streamedResponse = await request.send();
      final respStr = await streamedResponse.stream.bytesToString();

      if (streamedResponse.statusCode == 200) {
        setState(() {
          analysisResult = jsonDecode(respStr);
          isLoading = false;
        });
      } else {
        setState(() {
          errorMessage = 'Error ${streamedResponse.statusCode}: $respStr';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Exception: $e';
        isLoading = false;
      });
    }
  }

  Widget buildResultCard(Map<String, dynamic> data) {
    String clauseType = data['clause_type'] ?? 'Unknown';
    String riskLevel = data['risk_level'] ?? 'Unknown';
    String explanation = data['explanation'] ?? 'No explanation';
    String riskNote = data['risk_note'] ?? '';

    Color riskColor;
    IconData riskIcon;

    switch (riskLevel.toLowerCase()) {
      case 'low':
        riskColor = Colors.green;
        riskIcon = Icons.check_circle;
        break;
      case 'medium':
        riskColor = Colors.orange;
        riskIcon = Icons.warning;
        break;
      case 'high':
        riskColor = Colors.red;
        riskIcon = Icons.error;
        break;
      default:
        riskColor = Colors.grey;
        riskIcon = Icons.help;
    }

    return Card(
      elevation: 4,
      margin: EdgeInsets.symmetric(vertical: 10),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(riskIcon, color: riskColor),
                SizedBox(width: 10),
                Text(
                  clauseType,
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            SizedBox(height: 10),
            Text(
              'Risk Level: $riskLevel',
              style: TextStyle(color: riskColor, fontWeight: FontWeight.w600),
            ),
            SizedBox(height: 10),
            Text('Explanation:', style: TextStyle(fontWeight: FontWeight.bold)),
            Text(explanation),
            if (riskNote.isNotEmpty) ...[
              SizedBox(height: 10),
              Text('Risk Note:', style: TextStyle(fontWeight: FontWeight.bold)),
              Text(riskNote),
            ],
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Contract Analyzer')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: analyzeContract,
              child: Text('Upload Contract'),
            ),
            const SizedBox(height: 20),
            if (isLoading) CircularProgressIndicator(),
            if (errorMessage.isNotEmpty)
              Text(errorMessage, style: TextStyle(color: Colors.red)),
            if (analysisResult != null)
              Expanded(child: buildResultCard(analysisResult!)),
          ],
        ),
      ),
    );
  }
}
