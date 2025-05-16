    using System;
    using System.Collections.Generic;
    using System.ComponentModel.DataAnnotations;
    using System.ComponentModel.DataAnnotations.Schema;

    namespace BurnAnalysisApp.Models
    {
        public class PatientInfo
        {
            [Key]
            public int PatientID { get; set; }

            [Required]
            public string Name { get; set; } = string.Empty;
            public string? Email { get; set; }

            [Range(0, 120)]
            public int Age { get; set; }

            [Required]
            public string Gender { get; set; } = string.Empty;

            public string? MedicalHistory { get; set; }
            public string? BurnCause { get; set; }
            public string? BurnArea { get; set; }

            public DateTime? HospitalArrivalDate { get; set; }
            public DateTime? BurnOccurrenceDate { get; set; }
            public string? PhotoPath { get; set; }
            public string? BurnDepth { get; set; }
            public string? BurnPercentage { get; set; }  // Yeni özellik eklendi
            public string? AudioPath { get; set; } // Ses dosyasının yolunu saklamak için
            public bool ReminderSent { get; set; } = false; // Yeni alan
            public bool Verified { get; set; } = false;  // Yeni sütun
            public bool Trained { get; set; } = false;  //  Yeni alan
        }
    }
